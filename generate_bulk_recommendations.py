
import pandas as pd
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Data and Model Artifacts ---
logging.info("Loading data and model artifacts...")
try:
    df_cleaned = pd.read_csv('deployment_artifacts/cleaned_master_data.csv', low_memory=False)
    df_cleaned['Site ID'] = df_cleaned['Site ID'].astype(str)

    with open('deployment_artifacts/trained_svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)

    with open('deployment_artifacts/correlation_matrix.pkl', 'rb') as f:
        correlation_matrix = pickle.load(f)
        
    with open('deployment_artifacts/rebate_columns.pkl', 'rb') as f:
        rebate_columns = pickle.load(f)
    
    # Load the correct rebate name mapping from the same file used by the API
    with open('deployment_artifacts/rebate_name_mapping.pkl', 'rb') as f:
        rebate_name_map = pickle.load(f)
    
    logging.info(f"Loaded {len(df_cleaned)} customer records")
    logging.info(f"SVD model loaded successfully")
    logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    logging.info(f"Rebate columns: {rebate_columns}")
    logging.info(f"Rebate name mapping: {rebate_name_map}")
    
except FileNotFoundError as e:
    logging.error(f"CRITICAL ERROR: Missing data file - {e}. Cannot generate recommendations.")
    exit(1)
except Exception as e:
    logging.error(f"CRITICAL ERROR: Failed to load data - {e}. Cannot generate recommendations.")
    exit(1)

# --- Recommendation Functions (Replicated from FastAPI app) ---

def get_baseline_recommendations(site_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate baseline recommendations using popularity-based approach."""
    try:
        # Calculate global popularity
        global_popularity = {}
        for rebate in rebate_columns:
            if rebate in df_cleaned.columns:
                global_popularity[rebate] = df_cleaned[rebate].sum()
        
        # Sort by popularity
        popular_rebates = sorted(global_popularity.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for rebate, count in popular_rebates[:top_k]:
            recommendations.append({
                'rebate_code': rebate,
                'rebate_name': rebate_name_map.get(rebate, rebate),
                'score': round(count / max(global_popularity.values()), 3),
                'reason': f'Popular choice ({count} customers)',
                'model_used': 'baseline'
            })
        
        return recommendations
        
    except Exception as e:
        logging.error(f"Error in baseline recommendations: {e}")
        return []

def get_svd_recommendations(site_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate SVD collaborative filtering recommendations."""
    try:
        # Prepare data for SVD prediction
        user_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        if user_data.empty:
            return []
        
        # Get user's existing rebates (by rebate_name to handle duplicates)
        user_rebate_names = set()
        for rebate in rebate_columns:
            if rebate in user_data.columns and user_data[rebate].iloc[0] == 1:
                rebate_name = rebate_name_map.get(rebate, rebate)
                user_rebate_names.add(rebate_name)
        
        # Generate predictions for rebates user doesn't have
        recommendations = []
        seen_rebate_names = set()  # Track to avoid duplicates
        
        for rebate in rebate_columns:
            rebate_name = rebate_name_map.get(rebate, rebate)
            
            # Skip if user already has this rebate or we've already added this rebate_name
            if rebate_name not in user_rebate_names and rebate_name not in seen_rebate_names:
                try:
                    # Use SVD to predict rating for this rebate
                    prediction = svd_model.predict(site_id, rebate)
                    recommendations.append({
                        'rebate_code': rebate,
                        'rebate_name': rebate_name,
                        'score': round(prediction.est, 3),
                        'reason': 'Collaborative filtering prediction',
                        'model_used': 'svd'
                    })
                    seen_rebate_names.add(rebate_name)
                except Exception:
                    continue
        
        # Sort by score and return top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
        
    except Exception as e:
        logging.error(f"Error in SVD recommendations: {e}")
        return []

def get_hybrid_recommendations(site_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate hybrid recommendations combining SVD + correlation logic."""
    try:
        # Get SVD recommendations as base
        svd_recs = get_svd_recommendations(site_id, top_k * 2)  # Get more for hybrid processing
        
        # Get user's existing rebates for correlation boosting (by rebate_name to handle duplicates)
        user_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        if user_data.empty:
            return get_baseline_recommendations(site_id, top_k)
        
        user_rebates = []  # Keep original rebate codes for correlation matrix lookup
        user_rebate_names = set()  # Track rebate names to avoid recommending duplicates
        for rebate in rebate_columns:
            if rebate in user_data.columns and user_data[rebate].iloc[0] == 1:
                user_rebates.append(rebate)
                rebate_name = rebate_name_map.get(rebate, rebate)
                user_rebate_names.add(rebate_name)
        
        # Apply hybrid logic to SVD recommendations
        hybrid_recs = []
        for rec in svd_recs:
            rebate = rec['rebate_code']
            base_score = rec['score']
            reasons = [rec['reason']]
            
            # Apply correlation boosting/penalties with less aggressive penalties
            correlation_multiplier = 1.0
            correlation_bonus = 0
            for existing_rebate in user_rebates:
                if existing_rebate in correlation_matrix.index and rebate in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[existing_rebate, rebate]
                    
                    if correlation > 0.5:
                        correlation_bonus += 1.0  # Additive bonus for positive correlations
                        reasons.append(f"Correlates with {rebate_name_map.get(existing_rebate, existing_rebate)}")
                    elif correlation < -0.3:
                        correlation_multiplier *= 0.5  # Multiplicative penalty (reduces but doesn't zero)
                        reasons.append(f"Conflicts with {rebate_name_map.get(existing_rebate, existing_rebate)}")
            
            # Calculate final hybrid score (multiplicative penalty + additive bonus)
            hybrid_score = (base_score * correlation_multiplier) + correlation_bonus
            
            hybrid_recs.append({
                'rebate_code': rebate,
                'rebate_name': rec['rebate_name'],
                'score': round(max(0, hybrid_score), 3),  # Ensure non-negative
                'reason': '; '.join(reasons),
                'model_used': 'hybrid'
            })
        
        # Sort by hybrid score and return top_k
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_recs[:top_k]
        
    except Exception as e:
        logging.error(f"Error in hybrid recommendations: {e}")
        return get_baseline_recommendations(site_id, top_k)

# --- Main Bulk Generation Logic ---
logging.info("Starting bulk recommendation generation...")

# Get unique customers, filtering out NaN values
unique_customers = df_cleaned['Site ID'].dropna().unique()
unique_customers = [str(x) for x in unique_customers if not pd.isna(x)]
logging.info(f"Processing {len(unique_customers)} unique customers")

# Generate recommendations for all customers
all_recommendations = []
customer_count = 0

for site_id in unique_customers:
    customer_count += 1
    
    # Progress logging
    if customer_count % 1000 == 0:
        logging.info(f"Processed {customer_count}/{len(unique_customers)} customers")
    
    try:
        # Check if customer exists (they all should, but following API logic)
        customer_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        
        if not customer_data.empty:
            # Existing customer → Use HYBRID model
            recommendations = get_hybrid_recommendations(site_id, 5)
            customer_name = customer_data['Customer Name'].iloc[0] if 'Customer Name' in customer_data.columns else None
        else:
            # New customer → Use BASELINE model
            recommendations = get_baseline_recommendations(site_id, 5)
            customer_name = None
        
        # Ensure we always return recommendations
        if not recommendations:
            recommendations = get_baseline_recommendations(site_id, 5)
        
        # Add each recommendation as a row
        for rec in recommendations:
            all_recommendations.append({
                'site_id': site_id,
                'customer_name': customer_name,
                'rebate_code': rec['rebate_code'],
                'rebate_name': rec['rebate_name'],
                'score': rec['score'],
                'reason': rec['reason'],
                'model_used': rec['model_used']
            })
            
    except Exception as e:
        logging.error(f"Error processing customer {site_id}: {e}")
        continue

# Convert to DataFrame and save
logging.info(f"Generated {len(all_recommendations)} total recommendations")
df_recommendations = pd.DataFrame(all_recommendations)

# Save to CSV with proper formatting
output_file = 'all_recommendations.csv'
df_recommendations.to_csv(output_file, index=False, quoting=1, escapechar='\\')
logging.info(f"Saved recommendations to {output_file}")
logging.info(f"File size: {len(df_recommendations)} rows, {len(df_recommendations.columns)} columns")

print("\n=== BULK RECOMMENDATION GENERATION COMPLETE ===")
print(f"Total customers processed: {customer_count}")
print(f"Total recommendations generated: {len(all_recommendations)}")
print(f"Output file: {output_file}")
print(f"Average recommendations per customer: {len(all_recommendations) / customer_count:.1f}")
