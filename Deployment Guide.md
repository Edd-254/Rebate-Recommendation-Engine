# Deployment Guide: Rebate Recommendation Engine

This guide outlines the deployment strategy for the hybrid recommendation model and provides detailed specifications for the operational Power BI dashboard designed for Valley Water Conservation staff.

## Executive Summary

The Rebate Recommendation Engine is a hybrid AI system that combines collaborative filtering (SVD) with exploratory data analysis to provide personalized rebate recommendations for Valley Water customers. The system serves **all customers in the database** (15,189 records including 45 current 2025 customers) through an intuitive dashboard interface designed for daily staff operations and strategic program management.

---

## Dashboard Overview

The Power BI dashboard transforms our trained recommendation model into an operational tool that enables:
- **Proactive customer engagement** through personalized recommendations
- **Data-driven program management** through geographic and demographic analytics
- **Targeted marketing campaigns** through advanced customer filtering and export capabilities

The dashboard is structured into three main views, each serving different operational needs:

---

## Feature 1: Customer Lookup View

### **Purpose**
Daily operational tool for staff to look up any customer and receive AI-powered rebate recommendations.

### **Target Users**
Customer service representatives, field staff, program coordinators

### **Core Functionality**

#### **Customer Search & Profile**
- **Search Capabilities:** Find customers by Site ID, Address, or Customer Name
- **Customer Profile Display:** 
  - Site ID and basic demographics
  - City and Site Type (Single Family, Multi-Family, Commercial)
  - Complete rebate participation history
  - Current rebate status

#### **AI-Powered Recommendations**
- **Personalized Recommendations:** Top 3-5 rebate suggestions tailored to each customer
- **Recommendation Scoring:** Numerical scores (1-5) indicating recommendation confidence
- **Transparent Reasoning:** Clear explanations for why each rebate is recommended
- **Model Attribution:** Shows whether recommendation came from Hybrid SVD or Baseline model

### **Use Cases & Business Value**

**Scenario 1: Existing Customer Call**
- Staff receives call from customer interested in rebates
- Searches customer by name → sees they have irrigation rebates
- System recommends "Landscape Conversion" (high correlation)
- Staff explains: *"Based on your irrigation upgrades, landscape conversion would complement your water savings"*

**Scenario 2: New Customer Inquiry**
- New homeowner calls about available rebates
- Staff searches by address → no rebate history found
- System recommends "High-Efficiency Toilet" (most popular for Single Family in their city)
- Staff explains: *"This is our most popular rebate for homeowners in your area"*

**Scenario 3: Field Visit Follow-up**
- Inspector completes site visit, wants to suggest additional rebates
- Looks up customer → sees recent appliance rebate
- System recommends complementary landscape rebates
- Provides talking points for additional water-saving opportunities

---

## Feature 2: Geographic Recommendation Hotspots (Interactive Map)

### **Purpose**
Visual analysis of recommendation patterns across Valley Water's service area to inform resource allocation and marketing strategies.

### **Target Users**
Program managers, marketing team, executive leadership

### **Technical Implementation**
Interactive map using dynamic geocoding to convert city names and zip codes to precise latitude/longitude coordinates for accurate geographic visualization.

### **What Staff Will See**
- **Interactive Map Interface:** Valley Water service area with cities represented as colored points/areas
- **Visual Intensity:** Larger/darker areas indicate higher recommendation volumes
- **Click-through Details:** Click any city to see specific recommendation counts and rebate types
- **Real-time Data:** Reflects current recommendation patterns from the AI model

### **Use Cases & Business Value**

**Marketing Focus & Resource Allocation:**
- *"San Jose shows 500 pending recommendations - let's allocate more staff for outreach there"*
- *"Morgan Hill only has 12 recommendations - we need better program awareness campaigns"*
- *"Sunnyvale is generating lots of irrigation recommendations - ensure adequate rebate budget allocation"*

**Geographic Equity Analysis:**
- *"East San Jose shows very few recommendations - are we reaching underserved communities?"*
- *"Palo Alto has high engagement - what strategies can we replicate in other cities?"*

**Program Performance Monitoring:**
- *"Our new landscape rebate program is taking off in Cupertino - look at the recommendation density!"*
- *"Commercial properties in Santa Clara need attention - very low recommendation activity"*

---

## Feature 3: Recommendation Distribution by Site Type (Bar Chart)

### **Purpose**
Analyze recommendation patterns across different property types to ensure equitable program promotion and identify underserved segments.

### **Target Users**
Program managers, policy analysts, equity coordinators

### **Visual Design**
Horizontal or vertical bar chart comparing recommendation volumes across:
- Single Family Residential
- Multi-Family Residential 
- Commercial Properties
- Industrial Sites
- Other property types

### **What Staff Will See**
```
Single Family    ████████████████████ 1,200 recommendations (79%)
Multi-Family     ████████ 400 recommendations (16%)
Commercial       ███ 150 recommendations (4%)
Industrial       █ 50 recommendations (1%)
```

### **Use Cases & Business Value**

**Equity Analysis & Program Balance:**
- *"Single family homes get 80% of recommendations - are we neglecting apartment buildings?"*
- *"Multi-family properties represent 30% of our customers but only 16% of recommendations"*
- *"We need targeted outreach to apartment managers and property management companies"*

**Budget Planning & Resource Allocation:**
- *"Commercial properties generate fewer recommendations but higher rebate amounts per project"*
- *"Industrial sites rarely participate - maybe we need specialized rebate programs"*
- *"Adjust marketing budget allocation to match recommendation patterns"*

**Program Development Insights:**
- *"High single-family recommendations but low participation - are rebate amounts attractive enough?"*
- *"Commercial recommendations growing - create business-specific marketing materials"*
- *"Industrial segment needs different rebate structures - current programs don't fit their needs"*

---

## Feature 4: Rebate Correlation Heatmap

### **Purpose**
Visualize which rebate programs naturally complement each other to inform bundled marketing strategies and cross-selling opportunities.

### **Target Users**
Program managers, marketing team, customer service staff

### **Technical Foundation**
Based on the correlation matrix calculated from historical customer participation patterns, showing statistical relationships between different rebate programs.

### **Visual Design**
Color-coded matrix where:
- **Dark colors** = Strong positive correlation (rebates often taken together)
- **Light colors** = Weak correlation
- **Cool colors** = Negative correlation (rebates rarely taken together)

### **Use Cases & Business Value**

**Cross-selling Opportunities:**
- *"Customers with irrigation rebates have 85% correlation with landscape conversion - always mention both"*
- *"High-efficiency toilet and washing machine rebates are strongly correlated - create bundle offers"*

**Marketing Campaign Design:**
- *"Design 'Complete Home Water Efficiency' packages based on high-correlation rebate clusters"*
- *"Avoid promoting conflicting rebates in the same campaign (negative correlations)"*

**Staff Training & Talking Points:**
- *"When customer asks about sprinkler controllers, always mention drip irrigation conversion"*
- *"Rainwater capture and landscape rebates go hand-in-hand - explain the connection"*

---

## Feature 5: Customer Targeting & Outreach

### **Purpose**
Transform the dashboard from reactive tool to proactive marketing engine through advanced customer filtering and exportable campaign lists.

### **Target Users**
Marketing coordinators, outreach specialists, program managers

### **Advanced Filtering Capabilities**

#### **Geographic Filters:**
- City selection (San Jose, Palo Alto, Sunnyvale, etc.)
- Zip code ranges
- Service area boundaries

#### **Property Type Filters:**
- Single Family, Multi-Family, Commercial, Industrial
- Property size ranges
- Construction year periods

#### **Rebate Participation Filters:**
- **Has Rebates:** Customers who participated in specific programs
- **Missing Rebates:** Customers who haven't participated in specific programs
- **Recommendation Targets:** Customers for whom specific rebates are highly recommended

#### **Time-Based Filters:**
- Year of last participation (2020, 2021, 2022, etc.)
- Months since last interaction
- New customer identification (2025 participants)

### **Export Functionality**
Generate CSV files containing:
- Site ID and Customer Name
- Email Address (for direct outreach)
- City and Property Type
- Top Recommended Rebate
- Recommendation Reason
- Contact History

### **Use Cases & Business Value**

**Targeted Email Campaigns:**
- *"Export all San Jose single-family customers missing irrigation rebates"*
- *"Create campaign for 2023 participants who haven't engaged in 2024"*
- *"Target multi-family properties in Sunnyvale with landscape rebate recommendations"*

**Proactive Outreach Planning:**
- *"Identify 500 customers with high-confidence toilet rebate recommendations for phone campaign"*
- *"Find commercial properties with no recent participation for business development visits"*

**Performance Tracking:**
- *"Track conversion rates from targeted campaigns vs. general outreach"*
- *"Measure effectiveness of AI-driven recommendations in actual customer engagement"*

---

## Feature 6: Proactive Opportunities KPI

### **Purpose**
Quantify the number of customers with high-confidence recommendations available, providing a measurable metric for proactive engagement potential.

### **Target Users**
Executive leadership, program managers, performance analysts

### **Calculation Method**
Count of customers with recommendation scores above defined threshold (e.g., 4.0/5.0 or "High" confidence level)

### **Use Cases & Business Value**

**Performance Monitoring:**
- *"We have 1,200 high-confidence opportunities - our outreach team should contact 100 per week"*
- *"Proactive opportunities increased 15% this month - the model is finding more relevant matches"*

**Resource Planning:**
- *"High opportunity count justifies hiring additional outreach staff"*
- *"Low opportunities in certain areas indicate need for new rebate programs"*

**Executive Reporting:**
- *"AI model identified 2,500 proactive engagement opportunities worth $1.2M in potential rebates"*
- *"Conversion rate from high-confidence recommendations: 35% vs. 12% for general outreach"*
*   **Validates the Model:** The performance charts provide clear evidence of the recommendation engine's value to the organization.

---

## API Deployment Plan

To make the recommendation engine available to the Power BI dashboard and other potential applications, we will deploy it as a REST API using **FastAPI**. This approach provides a real-time, scalable, and maintainable solution.

### 1. Technology Stack

*   **API Framework:** **FastAPI** - Chosen for its high performance, automatic interactive documentation (Swagger UI), and modern Python features.
*   **Deployment Platform:** **Azure App Service** - A fully managed platform that simplifies deploying and scaling web applications without managing server infrastructure.
*   **Dependencies:** All necessary Python packages (`pandas`, `surprise`, etc.) will be listed in a `requirements.txt` file for easy installation on the server.

### 2. Required Files & Artifacts

## Deployment Requirements & Next Steps

### **Required Deployment Artifacts**

#### **Model Files (.pkl format)**
- `trained_svd_model.pkl` - The trained collaborative filtering model
- `correlation_matrix.pkl` - Pre-computed rebate correlation matrix
- `city_popularity.pkl` - Pre-computed city-level rebate popularity data
- `site_type_popularity.pkl` - Pre-computed site type rebate popularity data
- `rebate_name_mapping.pkl` - Human-readable rebate name mappings

#### **Data Files (.csv format)**
- `cleaned_master_data.csv` - Complete customer database (15,189 records)
- `city_coordinates.csv` - Geocoded city coordinates (generated during implementation)

#### **Application Files**
- `app.py` - Main FastAPI application with all endpoints
- `requirements.txt` - Python dependencies including geopy, fastapi, uvicorn, pandas, scikit-surprise

### **API Endpoint Specification**

#### **Core Recommendation Endpoints**
- `GET /recommendations/{site_id}` - AI recommendations for specific customer
- `GET /customer/{site_id}` - Customer profile and rebate history

#### **Analytics Endpoints for Dashboard**
- `GET /analytics/city-coordinates` - Geographic data for map visualization
- `GET /analytics/city-popularity` - City-level recommendation patterns
- `GET /analytics/site-type-popularity` - Property type recommendation patterns
- `GET /analytics/correlation-matrix` - Rebate correlation heatmap data
- `GET /analytics/dashboard-summary` - High-level KPIs and summary statistics

#### **Customer Targeting Endpoints**
- `GET /customers` - Filtered customer data with query parameters
- `GET /customers/export` - CSV export of filtered customer lists

### **Deployment Timeline**
- **Phase 1 (Week 1):** Model packaging and core API development
- **Phase 2 (Week 2):** Analytics endpoints and geocoding implementation
- **Phase 3 (Week 3):** Testing, deployment to Azure App Service, and Power BI integration

### **Security & Performance**
- API key authentication for secure access
- Rate limiting for production stability
- Caching for pre-computed analytics endpoints
- Error handling and logging for operational monitoring

---

## Model Integration & Recommendation Generation

### **Who Receives Recommendations**
The hybrid recommendation model serves **ALL customers** in the `cleaned_master_data.csv` database:
- **Historical customers** (2010-2024): 15,144 customers who trained the model
- **Current 2025 customers**: 45 newly added customers 
- **Future customers**: Any customers added to the system

### **Model Behavior by Customer Type**

#### **Existing Customers (Have Rebate History)**
- **Model Used:** Hybrid SVD + EDA-based boosting
- **Approach:** Analyzes past rebate participation to find similar customers and recommend complementary rebates
- **Example:** Customer with irrigation rebates → Model recommends landscape conversion (high correlation)
- **Dashboard Display:** Shows "Hybrid SVD" as model used, with specific correlation-based reasoning

#### **New Customers (No Rebate History)**  
- **Model Used:** Baseline model (EDA-based)
- **Approach:** Uses city and site type popularity to recommend most popular rebates for similar demographics
- **Example:** New single-family customer in San Jose → Model recommends high-efficiency toilet (most popular for that demographic)
- **Dashboard Display:** Shows "Baseline" as model used, with demographic-based reasoning

### **Real-time Recommendation Generation**
When staff searches for ANY customer in the dashboard:
1. **API receives customer Site ID**
2. **Model determines customer type** (existing vs. new)
3. **Appropriate algorithm generates recommendations** (Hybrid SVD vs. Baseline)
4. **Results returned with scores, reasons, and model attribution**
5. **Dashboard displays personalized recommendations** with full transparency

This ensures every customer in the system receives relevant, AI-powered recommendations regardless of their participation history.

---

## Technical Architecture & Data Flow

### **Hybrid API Design**
The system uses a hybrid approach combining pre-computed analytics for performance with raw data access for flexibility:

#### **Pre-computed Analytics Endpoints** (Fast Performance)
- `GET /analytics/city-popularity` - Pre-computed city rebate counts
- `GET /analytics/site-type-popularity` - Pre-computed site type analysis  
- `GET /analytics/correlation-matrix` - Pre-computed rebate correlations
- `GET /analytics/city-coordinates` - City data with geocoded coordinates
- `GET /analytics/dashboard-summary` - All key metrics in one call

#### **Raw Data Endpoints** (Maximum Flexibility)
- `GET /customers` - All customer records with filtering capabilities
- `GET /customers/{site_id}/recommendations` - Individual AI recommendations
- `GET /customers/{site_id}/history` - Customer rebate participation history
- `GET /customers/export` - Filtered customer lists as CSV downloads

#### **Query Parameters for Advanced Filtering**
- Geographic: `?city=San Jose&zip_code=95120`
- Property: `?site_type=Single Family`
- Rebate participation: `?has_rebates=irrigation,landscape&missing_rebates=toilet`
- Temporal: `?year_from=2023&year_to=2025`

### **Data Pipeline Integration**
The dashboard leverages our comprehensive data pipeline:
- **15,189 total customer records** including 45 current 2025 customers
- **Selective status expansion** for 2025 data (includes "Notice to Proceed Sent", "Post Insp. Approved")
- **Blank RebateType preservation** for 2025 customers
- **Real-time model integration** serving both historical and new customers

### **Model Deployment Architecture**
The trained hybrid model integrates seamlessly with the dashboard through:
1. **Packaged model artifacts** (.pkl files for SVD model, correlation matrix, popularity data)
2. **API model loading** functions for real-time recommendation generation
3. **Transparent model attribution** showing whether Hybrid SVD or Baseline model generated each recommendation
4. **Scalable recommendation serving** for all 15,189+ customers in the database

### **Dashboard Readiness Summary**

All confirmed dashboard features are technically feasible and have clear implementation paths:

✅ **Customer Lookup** - Fully supported by existing data and model
✅ **Geographic Hotspots Map** - Requires dynamic geocoding implementation
✅ **Site Type Distribution** - Fully supported by existing analytics
✅ **Correlation Heatmap** - Fully supported by existing correlation matrix
✅ **Customer Targeting & Export** - Requires API filtering and export endpoints
✅ **Proactive Opportunities KPI** - Fully supported by model output

The dashboard will transform our trained AI model into an operational tool that serves all 15,189+ customers in the database, enabling both reactive customer service and proactive marketing campaigns.

---

## Implementation Guide

This section provides detailed technical implementation steps for each dashboard feature.

### 1. Geographic Recommendation Hotspots (Map)

**Implementation Approach: Dynamic Geocoding**

**Step 1: Add Geocoding Dependencies**
```bash
# Add to requirements.txt
geopy==2.4.1
```

**Step 2: Implement Geocoding Function**
```python
# Add to build_notebook.py or separate geocoding module
from geopy.geocoders import Nominatim
import time

def geocode_location(city_name, zip_code=None):
    """Convert city name/zip to latitude/longitude coordinates"""
    geolocator = Nominatim(user_agent="valley_water_rebates")
    
    # Try city first, then zip as fallback
    query = f"{city_name}, CA, USA"
    if zip_code and zip_code != '00nan':
        query = f"{zip_code}, CA, USA"
    
    try:
        location = geolocator.geocode(query)
        time.sleep(1)  # Rate limiting for free tier
        return (location.latitude, location.longitude) if location else None
    except Exception as e:
        print(f"Geocoding failed for {query}: {e}")
        return None

def generate_city_coordinates():
    """Generate coordinates for all cities in dataset"""
    df = pd.read_csv('cleaned_master_data.csv')
    unique_cities = df['City'].dropna().unique()
    
    coordinates_data = []
    for city in unique_cities:
        coords = geocode_location(city)
        if coords:
            coordinates_data.append({
                'city': city,
                'latitude': coords[0],
                'longitude': coords[1]
            })
    
    coords_df = pd.DataFrame(coordinates_data)
    coords_df.to_csv('city_coordinates.csv', index=False)
    return coords_df
```

**Step 3: API Endpoint for Coordinates**
```python
# In FastAPI app
@app.get("/analytics/city-coordinates")
def get_city_coordinates():
    """Return city popularity data with coordinates for mapping"""
    # Load city popularity and coordinates
    city_pop = pd.read_csv('city_popularity.csv')  # Pre-computed
    city_coords = pd.read_csv('city_coordinates.csv')
    
    # Merge data
    map_data = city_pop.merge(city_coords, on='city', how='left')
    return map_data.to_dict('records')
```

### 2. Customer Filtering and Export

**Implementation Approach: API Endpoints with Pandas Filtering**

**Step 1: Filtering Logic**
```python
def filter_customers(city=None, site_type=None, has_rebates=None, 
                    missing_rebates=None, year_from=None, year_to=None):
    """Filter customers based on multiple criteria"""
    df = pd.read_csv('cleaned_master_data.csv')
    df['Request Date'] = pd.to_datetime(df['Request Date'])
    
    # Apply filters
    if city:
        df = df[df['City'].isin(city if isinstance(city, list) else [city])]
    
    if site_type:
        df = df[df['Site Type'].isin(site_type if isinstance(site_type, list) else [site_type])]
    
    if has_rebates:
        for rebate in has_rebates:
            rebate_col = f'has_{rebate.lower().replace(" ", "_")}_rebate'
            if rebate_col in df.columns:
                df = df[df[rebate_col] == True]
    
    if missing_rebates:
        for rebate in missing_rebates:
            rebate_col = f'has_{rebate.lower().replace(" ", "_")}_rebate'
            if rebate_col in df.columns:
                df = df[df[rebate_col] == False]
    
    if year_from:
        df = df[df['Request Date'].dt.year >= int(year_from)]
    
    if year_to:
        df = df[df['Request Date'].dt.year <= int(year_to)]
    
    return df
```

**Step 2: Export Functionality**
```python
from fastapi.responses import StreamingResponse
import io

@app.get("/customers/export")
def export_customers(city: str = None, site_type: str = None, 
                    has_rebates: str = None, missing_rebates: str = None,
                    year_from: int = None, year_to: int = None):
    """Export filtered customer list as CSV"""
    
    # Parse comma-separated lists
    has_rebates_list = has_rebates.split(',') if has_rebates else None
    missing_rebates_list = missing_rebates.split(',') if missing_rebates else None
    
    # Filter customers
    filtered_df = filter_customers(
        city=city, site_type=site_type,
        has_rebates=has_rebates_list, missing_rebates=missing_rebates_list,
        year_from=year_from, year_to=year_to
    )
    
    # Add top recommendations for each customer
    export_data = []
    for _, customer in filtered_df.iterrows():
        recommendations = get_recommendations(customer['Site ID'])
        top_rec = recommendations[0] if recommendations else None
        
        export_data.append({
            'Site ID': customer['Site ID'],
            'Customer Name': customer['Customer Name'],
            'Email Address': customer['Email Address'],
            'City': customer['City'],
            'Site Type': customer['Site Type'],
            'Top Recommended Rebate': top_rec['rebate'] if top_rec else 'None',
            'Recommendation Reason': top_rec['reason'] if top_rec else 'No recommendations'
        })
    
    # Convert to CSV
    export_df = pd.DataFrame(export_data)
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=customer_export.csv"}
    )
```

**Step 3: Raw Data API for Power BI**
```python
@app.get("/customers")
def get_customers(city: str = None, site_type: str = None,
                 has_rebates: str = None, missing_rebates: str = None,
                 year_from: int = None, year_to: int = None):
    """Get filtered customer data for Power BI consumption"""
    
    # Same filtering logic as export
    filtered_df = filter_customers(
        city=city, site_type=site_type,
        has_rebates=has_rebates.split(',') if has_rebates else None,
        missing_rebates=missing_rebates.split(',') if missing_rebates else None,
        year_from=year_from, year_to=year_to
    )
    
    return filtered_df.to_dict('records')
```

### 3. Pre-computed Analytics Endpoints

**Implementation: Serve Existing Calculated Data**

```python
@app.get("/analytics/city-popularity")
def get_city_popularity():
    """Return pre-computed city popularity data"""
    # This data is already calculated in notebook
    city_pop = pd.read_csv('city_popularity.csv')  # Generated by notebook
    return city_pop.to_dict('records')

@app.get("/analytics/site-type-popularity")
def get_site_type_popularity():
    """Return pre-computed site type popularity data"""
    site_pop = pd.read_csv('site_type_popularity.csv')  # Generated by notebook
    return site_pop.to_dict('records')

@app.get("/analytics/correlation-matrix")
def get_correlation_matrix():
    """Return rebate correlation matrix for heatmap"""
    corr_matrix = pd.read_csv('correlation_matrix.csv', index_col=0)
    return corr_matrix.to_dict('index')
```

### Implementation Timeline

**Phase 1 (Week 1): Core API Development**
- Create FastAPI application structure
- Implement customer lookup and recommendation endpoints
- Add basic filtering functionality

**Phase 2 (Week 2): Analytics and Export**
- Implement geocoding functionality
- Add pre-computed analytics endpoints
- Implement CSV export functionality

**Phase 3 (Week 3): Testing and Deployment**
- Test all API endpoints
- Deploy to Azure App Service
- Create Power BI dashboard connections

**Total Estimated Timeline: 3 weeks**
