"""
Rebate Recommendation Engine API
Valley Water Conservation District

FastAPI application providing AI-powered rebate recommendations
for internal staff use via Power BI dashboard integration.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rebate Recommendation Engine API",
    description="AI-powered rebate recommendations for Valley Water Conservation District",
    version="1.0.0"
)

# Global variables for loaded data
svd_model = None
correlation_matrix = None
rebate_name_map = None
city_popularity_df = None
site_type_popularity_df = None
dashboard_summary = None
df_cleaned = None
city_coordinates_df = None
rebate_columns = None
unique_cities = None
unique_site_types = None

@app.on_event("startup")
async def load_models_and_data():
    """Load all models and data at application startup."""
    global svd_model, correlation_matrix, rebate_name_map, city_popularity_df
    global site_type_popularity_df, dashboard_summary, df_cleaned, city_coordinates_df
    global rebate_columns, unique_cities, unique_site_types
    
    try:
        logger.info("Loading deployment artifacts...")
        
        # Load models
        with open('deployment_artifacts/trained_svd_model.pkl', 'rb') as f:
            svd_model = pickle.load(f)
        logger.info("✓ Loaded SVD model")
        
        # Load supporting data
        with open('deployment_artifacts/correlation_matrix.pkl', 'rb') as f:
            correlation_matrix = pickle.load(f)
        logger.info("✓ Loaded correlation matrix")
        
        with open('deployment_artifacts/rebate_name_mapping.pkl', 'rb') as f:
            rebate_name_map = pickle.load(f)
        logger.info("✓ Loaded rebate name mapping")
        
        # Load analytics data
        with open('deployment_artifacts/city_popularity.pkl', 'rb') as f:
            city_popularity_df = pickle.load(f)
        logger.info("✓ Loaded city popularity data")
        
        with open('deployment_artifacts/site_type_popularity.pkl', 'rb') as f:
            site_type_popularity_df = pickle.load(f)
        logger.info("✓ Loaded site type popularity data")
        
        with open('deployment_artifacts/dashboard_summary.pkl', 'rb') as f:
            dashboard_summary = pickle.load(f)
        logger.info("✓ Loaded dashboard summary")
        
        # Load supporting data for filtering
        with open('deployment_artifacts/rebate_columns.pkl', 'rb') as f:
            rebate_columns = pickle.load(f)
        logger.info("✓ Loaded rebate columns")
        
        with open('deployment_artifacts/unique_cities.pkl', 'rb') as f:
            unique_cities = pickle.load(f)
        logger.info("✓ Loaded unique cities")
        
        with open('deployment_artifacts/unique_site_types.pkl', 'rb') as f:
            unique_site_types = pickle.load(f)
        logger.info("✓ Loaded unique site types")
        
        # Load customer data
        df_cleaned = pd.read_csv('deployment_artifacts/cleaned_master_data.csv')
        logger.info(f"✓ Loaded customer data ({len(df_cleaned)} records)")
        
        # Load city coordinates
        city_coordinates_df = pd.read_csv('deployment_artifacts/city_coordinates.csv')
        logger.info(f"✓ Loaded city coordinates ({len(city_coordinates_df)} cities)")
        
        logger.info("All deployment artifacts loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load deployment artifacts: {e}")
        raise e

# === CORE RECOMMENDATION FUNCTIONS ===

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
        logger.error(f"Error in baseline recommendations: {e}")
        return []

def get_svd_recommendations(site_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate SVD collaborative filtering recommendations."""
    try:
        from surprise import Dataset, Reader
        
        # Prepare data for SVD prediction
        # Create user-item matrix for this user
        user_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        if user_data.empty:
            return []
        
        # Get user's existing rebates
        user_rebates = []
        for rebate in rebate_columns:
            if rebate in user_data.columns and user_data[rebate].iloc[0] == 1:
                user_rebates.append(rebate)
        
        # Generate predictions for rebates user doesn't have
        recommendations = []
        for rebate in rebate_columns:
            if rebate not in user_rebates:
                try:
                    # Use SVD to predict rating for this rebate
                    prediction = svd_model.predict(site_id, rebate)
                    recommendations.append({
                        'rebate_code': rebate,
                        'rebate_name': rebate_name_map.get(rebate, rebate),
                        'score': round(prediction.est, 3),
                        'reason': 'Collaborative filtering prediction',
                        'model_used': 'svd'
                    })
                except Exception:
                    continue
        
        # Sort by score and return top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
        
    except Exception as e:
        logger.error(f"Error in SVD recommendations: {e}")
        return []

def get_hybrid_recommendations(site_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate hybrid recommendations combining SVD + correlation logic."""
    try:
        # Get SVD recommendations as base
        svd_recs = get_svd_recommendations(site_id, top_k * 2)  # Get more for hybrid processing
        
        # Get user's existing rebates for correlation boosting
        user_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        if user_data.empty:
            return get_baseline_recommendations(site_id, top_k)
        
        user_rebates = []
        for rebate in rebate_columns:
            if rebate in user_data.columns and user_data[rebate].iloc[0] == 1:
                user_rebates.append(rebate)
        
        # Apply hybrid logic to SVD recommendations
        hybrid_recs = []
        for rec in svd_recs:
            rebate = rec['rebate_code']
            base_score = rec['score']
            reasons = [rec['reason']]
            
            # Apply correlation boosting/penalties
            correlation_boost = 0
            for existing_rebate in user_rebates:
                if existing_rebate in correlation_matrix.index and rebate in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[existing_rebate, rebate]
                    
                    if correlation > 0.5:
                        correlation_boost += 2.0
                        reasons.append(f"Correlates with {rebate_name_map.get(existing_rebate, existing_rebate)}")
                    elif correlation < -0.3:
                        correlation_boost -= 1.5
                        reasons.append(f"Conflicts with {rebate_name_map.get(existing_rebate, existing_rebate)}")
            
            # Calculate final hybrid score
            hybrid_score = base_score + correlation_boost
            
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
        logger.error(f"Error in hybrid recommendations: {e}")
        return get_baseline_recommendations(site_id, top_k)

# === API ENDPOINTS ===

@app.get("/")
async def root():
    """API health check and information."""
    return {
        "message": "Rebate Recommendation Engine API",
        "version": "1.0.0",
        "status": "active",
        "total_customers": len(df_cleaned) if df_cleaned is not None else 0
    }

@app.get("/recommendations/{site_id}")
async def get_recommendations(
    site_id: str,
    top_k: int = Query(5, ge=1, le=10, description="Number of recommendations to return")
):
    """
    Get AI-powered rebate recommendations for a customer.
    
    Uses hybrid model for existing customers, baseline for new customers.
    """
    try:
        # Check if customer exists
        customer_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        
        if not customer_data.empty:
            # Existing customer → Use HYBRID model
            recommendations = get_hybrid_recommendations(site_id, top_k)
            customer_name = customer_data['Customer Name'].iloc[0] if 'Customer Name' in customer_data.columns else None
        else:
            # New customer → Use BASELINE model
            recommendations = get_baseline_recommendations(site_id, top_k)
            customer_name = None
        
        # Ensure we always return recommendations
        if not recommendations:
            recommendations = get_baseline_recommendations(site_id, top_k)
        
        return {
            "site_id": site_id,
            "customer_name": customer_name,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations for {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/customer/{site_id}")
async def get_customer_profile(site_id: str):
    """Get customer profile and rebate history."""
    try:
        customer_data = df_cleaned[df_cleaned['Site ID'] == site_id]
        
        if customer_data.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get customer info
        customer_info = customer_data.iloc[0]
        
        # Get rebate history
        rebate_history = []
        for rebate in rebate_columns:
            if rebate in customer_data.columns and customer_info[rebate] == 1:
                rebate_history.append({
                    'rebate_code': rebate,
                    'rebate_name': rebate_name_map.get(rebate, rebate)
                })
        
        return {
            "site_id": site_id,
            "customer_name": customer_info.get('Customer Name'),
            "city": customer_info.get('City'),
            "site_type": customer_info.get('Site Type'),
            "rebate_history": rebate_history,
            "total_rebates": len(rebate_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer profile for {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === ANALYTICS ENDPOINTS FOR DASHBOARD ===

@app.get("/analytics/city-coordinates")
async def get_city_coordinates():
    """Get geocoded city coordinates for map visualization."""
    try:
        return city_coordinates_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting city coordinates: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/city-popularity")
async def get_city_popularity():
    """Get city-level rebate popularity statistics."""
    try:
        return city_popularity_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting city popularity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/site-type-popularity")
async def get_site_type_popularity():
    """Get site type rebate popularity statistics."""
    try:
        return site_type_popularity_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error getting site type popularity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/correlation-matrix")
async def get_correlation_matrix():
    """Get rebate correlation matrix for heatmap visualization."""
    try:
        # Convert correlation matrix to format suitable for heatmap
        correlation_data = []
        for rebate1 in correlation_matrix.index:
            for rebate2 in correlation_matrix.columns:
                correlation_data.append({
                    'rebate1': rebate_name_map.get(rebate1, rebate1),
                    'rebate2': rebate_name_map.get(rebate2, rebate2),
                    'correlation': round(correlation_matrix.loc[rebate1, rebate2], 3)
                })
        
        return correlation_data
    except Exception as e:
        logger.error(f"Error getting correlation matrix: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/dashboard-summary")
async def get_dashboard_summary():
    """Get high-level KPIs and summary statistics."""
    try:
        return dashboard_summary
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === CUSTOMER TARGETING ENDPOINTS ===

@app.get("/customers")
async def get_filtered_customers(
    city: Optional[str] = Query(None, description="Filter by city"),
    site_type: Optional[str] = Query(None, description="Filter by site type"),
    has_rebate: Optional[str] = Query(None, description="Filter customers who have this rebate"),
    missing_rebate: Optional[str] = Query(None, description="Filter customers missing this rebate")
):
    """Get filtered customer data for targeting and outreach."""
    try:
        filtered_df = df_cleaned.copy()
        
        # Apply filters
        if city:
            filtered_df = filtered_df[filtered_df['City'] == city]
        
        if site_type:
            filtered_df = filtered_df[filtered_df['Site Type'] == site_type]
        
        if has_rebate and has_rebate in rebate_columns:
            filtered_df = filtered_df[filtered_df[has_rebate] == 1]
        
        if missing_rebate and missing_rebate in rebate_columns:
            filtered_df = filtered_df[filtered_df[missing_rebate] == 0]
        
        # Get total count before formatting
        total_count = len(filtered_df)
        
        # Format response
        customers = []
        for _, customer in filtered_df.iterrows():
            customers.append({
                'site_id': customer['Site ID'],
                'customer_name': customer.get('Customer Name'),
                'city': customer.get('City'),
                'site_type': customer.get('Site Type'),
                'email': customer.get('Email Address')
            })
        
        return {
            "customers": customers,
            "total_count": total_count,
            "filters_applied": {
                "city": city,
                "site_type": site_type,
                "has_rebate": has_rebate,
                "missing_rebate": missing_rebate
            }
        }
        
    except Exception as e:
        logger.error(f"Error filtering customers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/customers/export")
async def export_filtered_customers(
    city: Optional[str] = Query(None, description="Filter by city"),
    site_type: Optional[str] = Query(None, description="Filter by site type"),
    has_rebate: Optional[str] = Query(None, description="Filter customers who have this rebate"),
    missing_rebate: Optional[str] = Query(None, description="Filter customers missing this rebate")
):
    """Export filtered customer data as CSV for campaigns."""
    try:
        filtered_df = df_cleaned.copy()
        
        # Apply same filters as /customers endpoint
        if city:
            filtered_df = filtered_df[filtered_df['City'] == city]
        
        if site_type:
            filtered_df = filtered_df[filtered_df['Site Type'] == site_type]
        
        if has_rebate and has_rebate in rebate_columns:
            filtered_df = filtered_df[filtered_df[has_rebate] == 1]
        
        if missing_rebate and missing_rebate in rebate_columns:
            filtered_df = filtered_df[filtered_df[missing_rebate] == 0]
        
        # Select export columns
        export_columns = ['Site ID', 'Customer Name', 'City', 'Site Type', 'Email Address']
        export_df = filtered_df[export_columns]
        
        # Generate CSV
        output = io.StringIO()
        export_df.to_csv(output, index=False)
        output.seek(0)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"customer_export_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error exporting customers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
