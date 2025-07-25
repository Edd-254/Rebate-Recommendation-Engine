# Rebate Program Recommendation Engine:# Project Implementation

## Problem Understanding and Proposed Solution

### The Problem: An Internal Staff Perspective

[...existing content remains unchanged...]

### The Problem: An Internal Staff Perspective

From the perspective of the Valley Water Conservation unit staff, the current process for advising customers on rebates is inefficient and lacks personalization. Key challenges include:

1. **Reactive vs. Proactive Service:** Staff often have to wait for customers to inquire about rebates. There is no efficient system to proactively identify and contact customers who are strong candidates for specific programs.
2. **Information Silos:** To provide tailored advice, staff may need to manually cross-reference multiple datasets or rely on their own knowledge, which is time-consuming and can lead to inconsistent or incomplete recommendations.
3. **Missed Opportunities:** Without a clear, data-driven view of a customer's needs and history, staff may miss opportunities to recommend additional, relevant rebates that a customer would be eligible for, limiting both water savings and customer satisfaction.

This operational friction makes it difficult for staff to maximize their impact and drive adoption of key conservation programs.

### The Proposed Solution: An Internal Recommendation Dashboard

The goal of this project is to build a **Rebate Program Recommendation Engine** delivered through an **internal dashboard**. This tool will empower staff by automatically generating tailored rebate suggestions for any customer.

Instead of manually researching, staff can instantly see a ranked list of the most relevant rebates for a specific customer, turning every interaction into an opportunity for proactive, personalized engagement.

### Core Use Case: The Internal Dashboard for Staff

This dashboard is the central tool for staff to enhance customer service and drive rebate adoption. It supports workflows for both new and existing customers.

**A. Staff Workflow for a New Customer**

1. **Customer Lookup or Entry:** Staff can search for an existing customer or enter basic profile info for a new one (name, address, site type, etc.).
2. **Profile Creation:** The system creates a profile for the new customer, even with no rebate history.
3. **Recommendation Generation:** The engine instantly generates recommendations based on what’s popular for similar customers and properties.
4. **Staff Action:** Staff can review the recommendations with the customer, explaining the reasons provided by the system (e.g., “Popular for single-family homes in your zip code”).

**B. Staff Workflow for an Existing Customer**

1. **Customer Search:** Staff can look up any current customer by name, Site ID, or address.
2. **Customer Profile View:** The dashboard shows the customer’s complete rebate history.
3. **Cross-Rebate Recommendations:** The system highlights additional rebates the customer hasn’t used but is likely to benefit from, based on their profile and the behavior of similar customers.
4. **Proactive Outreach:** Staff can use these insights to suggest new rebates during customer interactions.

**Summary Table**

| Staff Action                       | New Customer | Existing Customer |
| ---------------------------------- | ------------ | ----------------- |
| Search/Add Customer                | Yes          | Yes               |
| View Profile & History             | Yes (empty)  | Yes (full)        |
| See Recommended Rebates            | Yes          | Yes               |
| See Reason for Recommendations     | Yes          | Yes               |
| Recommend via Call/Email/In-Person | Yes          | Yes               |

**Key Benefits**

- **Personalized Service:** Staff can provide tailored, data-driven advice, not generic lists.
- **Proactive Engagement:** Staff are empowered to reach out to customers who are likely to benefit from additional rebates.
- **Efficiency:** Staff can provide high-quality recommendations for new and existing customers in a fraction of the time.

---

## Data Exploration and Validation Findings

Our initial exploration has validated key assumptions about the data:

- **Date Range**: Application List.csv contains data from 2015-07-01 to present (with some erroneous future dates)
- **Rebate Status Values**: Multiple valid statuses identified including "Past Rebate", "Check Issued", "Ready to be Paid", etc.
- **Dataset Overlap**: We confirmed that 15,503 approved/completed rebate numbers in Application List.csv can be matched with 7,772 rebate numbers in Appl.Hardware.csv
- **Sample Overlapping Rebate Numbers**: '29988', '28388', '9788', '50827', '11842', '46885', '53105', '56891', '48624', '6788'
- **Dashboard Readiness Update (July 23, 2025):** The data pipeline now includes `Customer Name` and `Email Address` in `cleaned_master_data.csv`. All dashboard export and outreach features are fully supported.

This confirms our ability to join the datasets and proceed with a collaborative filtering approach as initially proposed.

## Project Phases and Implementation Plan

### Phase 1: Data Preparation and Exploration

- [X] Load and examine raw data files
- [X] Validate relationship between Application List.csv and Appl.Hardware.csv
- [X] Confirm data join feasibility via Rebate Number
- [X] Clean, standardize, and join data to create master dataset

  - [X] Filter for approved/completed applications
  - [X] Handle missing values (e.g., coercing dates)
  - [X] Standardize city names and zip codes
  - [X] Convert dates to proper format
  - [X] Create unified master dataset (`cleaned_master_data.csv`)
- [ ] **Perform Exploratory Data Analysis (EDA).** The objective of this phase is to move from a clean dataset to actionable insights. By exploring the data, we will validate our core assumptions, uncover hidden patterns in customer behavior, and build a strong, evidence-based foundation for the recommendation model. The analysis will focus on four key areas:

  - **1. Rebate Program Popularity and Distribution**

    - **Why it's important:** Understanding the baseline popularity of each rebate is essential for building a robust "cold-start" recommendation strategy for new customers. It also helps us distinguish between mainstream programs and niche ones that may require more targeted promotion.
    - **Actions to be taken:**
      - Calculate and visualize the total participation counts for each of the five landscape sub-rebates and the graywater program.
      - Generate summary statistics to quantify the reach of each program.
  - **2. Geographic Pattern Analysis**

    - **Why it's important:** The project hypothesizes that location is a key factor in rebate adoption. This analysis will validate that assumption by identifying geographic "hotspots." These insights will power the geographic similarity component of our hybrid model, allowing us to make recommendations based on neighborhood-level trends.
    - **Actions to be taken:**
      - Analyze the distribution of rebate applications across different cities and zip codes.
      - Create visualizations to highlight the top 10-20 cities with the highest rebate participation.
  - **3. Rebate Co-occurrence and Correlation**

    - **Why it's important:** This analysis is the foundation of our collaborative filtering approach ("people who adopted X also adopted Y"). By identifying which rebates are frequently adopted together, we can discover natural "bundles" and create powerful, data-driven recommendation rules.
    - **Actions to be taken:**
      - Create a co-occurrence matrix to see which pairs of rebates are most frequently taken by the same customer.
      - Visualize these correlations to make the relationships easy to interpret.
  - **4. Analysis by Customer Segment (Site Type)**

    - **Why it's important:** To deliver on the promise of personalized advice, the model must understand that different customers have different needs. By analyzing preferences based on `Site Type`, we ensure the engine recommends relevant programs (e.g., not suggesting a single-family home rebate to a large commercial property).
    - **Actions to be taken:**
      - Segment the dataset by `Site Type` (e.g., 'Single Family', 'Commercial', 'MFD').
      - Compare the popularity of different rebates within each segment to learn the unique preferences of each group.

### Phase 2 Findings: Exploratory Data Analysis Insights

The Exploratory Data Analysis (EDA) provided critical, data-driven insights that directly validate our proposed hybrid recommendation model. The findings confirm that a combination of baseline popularity, geographic trends, rebate co-occurrence, and customer segmentation will be highly effective.

Here is a summary of the key findings and their direct implications for the model's design:

#### 1. Rebate Popularity: A Clear Hierarchy

* **Key Finding:** The **Landscape Conversion** rebate is overwhelmingly the most popular program, with over 10,000 participants. The **WBIC** and **Irrigation Upgrade** rebates also show strong participation. In contrast, the Graywater, Rainwater Capture, and Drip Conversion programs are far more niche.
* **Implication for the Model:** This gives us a strong baseline. For new customers where we have little information (a "cold start"), recommending the popular Landscape Conversion rebate is a safe and effective default strategy.

#### 2. Geographic Patterns: Location is a Key Predictor

* **Key Finding:** Rebate adoption is heavily concentrated geographically. **San Jose** is the clear leader with more than five times the applications of the next closest city, Sunnyvale.
* **Implication for the Model:** This strongly validates our hypothesis that geographic similarity is a powerful predictor. Our model will use a customer's city and zip code to inform recommendations based on local trends.

#### 3. Rebate Co-occurrence: Powerful Recommendation Rules

* **Key Finding:** The correlation matrix revealed two critical, non-obvious relationships:
  1. **Strong Positive Correlation (+0.57):** `WBIC` and `Irrigation Upgrade` rebates are frequently adopted together.
  2. **Strong Negative Correlation (-0.63):** `Landscape Conversion` is negatively correlated with both `WBIC` and `Irrigation Upgrades`. This suggests they are often mutually exclusive.
* **Implication for the Model:** This is the core of our collaborative filtering logic. We can now create powerful rules:
  * **Rule 1 (Bundling):** If a user adopts a `WBIC` rebate, recommend an `Irrigation Upgrade` (and vice-versa).
  * **Rule 2 (Exclusion):** If a user adopts a `Landscape Conversion` rebate, do *not* recommend a `WBIC` or `Irrigation Upgrade` rebate, as it's likely redundant.

#### 4. Site Type Analysis: The Importance of Segmentation

* **Key Finding:** **Single Family** homes are the dominant participants across all rebate categories. Furthermore, some programs, like Graywater and Rainwater Capture, are almost exclusively adopted by single-family residences.
* **Implication for the Model:** This confirms that a one-size-fits-all approach will not work. The model must be segmented by `Site Type` to ensure it makes contextually relevant recommendations (e.g., not suggesting a residential-only rebate to a commercial property).

### Data Integration Strategy: Direct Identifier Matching

The initial and most direct approach to integrating the graywater rebate data was to identify a common, unique identifier between the `cleaned_master_data.csv` (containing landscape rebates) and the `Graywater Rebate.csv` datasets. Based on an analysis of available columns, two potential keys were identified:

- **`Rebate Number`**: The standard identifier used in the landscape program.
- **`LRP Rebate Number`**: A column in the graywater data that appeared to reference the Landscape Rebate Program.

The methodology involved cleaning and standardizing these columns in both datasets and then performing a direct intersection to find matching values. The analysis conclusively showed **zero** overlapping records between the two programs using either key. This result demonstrated that a simple, direct join was not feasible and that a more advanced technique would be required to link customers across the different rebate programs.

### Data Integration Strategy: Fuzzy Matching for Graywater Data

After initial investigations revealed no direct, reliable join key (e.g., `Rebate Number` or `LRP Rebate Number`) between the landscape and graywater rebate datasets, a more sophisticated approach was required to identify customers participating in both programs. The following fuzzy matching methodology was implemented to link records based on customer and site information.

**1. Rationale:**
Direct identifiers proved unreliable, with zero matches found. This is common in real-world datasets where different programs may use different numbering schemes or where data entry practices vary. Fuzzy matching overcomes these limitations by identifying non-exact matches based on string similarity, making it ideal for linking records via customer names and addresses, which may contain minor typos, abbreviations, or formatting differences.

**2. Methodology:**
To create a robust basis for comparison, a combined 'match string' was created for each record in both datasets. This string concatenates standardized customer name and address information.

- **Source Files:**
  - `Application List.csv` (for landscape rebates)
  - `Graywater Rebate.csv` (for graywater rebates)
- **Columns Used for Matching:**
  - From `Application List.csv`: `Customer Name`, `Address/Site`
  - From `Graywater Rebate.csv`: `Customer Name`, `Installation Address`
- **Standardization Process:** Before matching, all values from these columns were converted to lowercase and stripped of leading/trailing whitespace to ensure consistency.

**3. Technology and Scoring:**

- **Library:** The `thefuzz` Python library (formerly `fuzzywuzzy`) was used for this task.
- **Scoring Algorithm:** The `fuzz.token_sort_ratio` scorer was selected. This method tokenizes the strings, sorts the tokens alphabetically, and then calculates a similarity score. It is highly effective at handling cases where the word order is different (e.g., "John Smith 123 Main St" vs. "123 Main St John Smith").

**4. Matching Criteria:**

- A high confidence **similarity threshold of 80%** was established. For each record in the graywater dataset, the process searches for the best possible match in the entire landscape dataset. Only if the top match has a score of 90 or greater is it considered a confirmed link between the two programs. This strict threshold minimizes the risk of false positives.

This documented approach ensures that the integration of graywater data is both transparent and reproducible, providing a solid foundation for subsequent analysis and model development.

### Phase 2: Feature Engineering

This phase focuses on transforming the raw, cleaned data into meaningful features that directly power the recommendation engine. The feature engineering for the baseline model is complete and focused on creating a clear, interpretable user-item interaction dataset.

**Completed Feature Engineering:**

- **[X] Created User-Item Interaction Data:** The core of our recommendation data was created by transforming the `RebateType` column. This single, composite column was deconstructed into five distinct boolean features, one for each sub-rebate (e.g., `has_landscape_conversion_rebate`, `has_wbic_rebate`, etc.). These columns now serve as our primary user-item interaction data, indicating which customers (users) have participated in which rebate programs (items).
- **[X] Integrated External Data via Fuzzy Matching:** A critical feature, `participated_in_graywater`, was engineered by integrating the separate `Graywater Rebate.csv` dataset. Since no reliable direct key existed, this was achieved using a sophisticated fuzzy matching algorithm based on customer name and address, with a high confidence threshold to ensure data quality. This added a sixth key rebate program to our user-item dataset.
- **[X] Prepared Key Categorical Features:** The following categorical features, identified as critical during EDA, have been cleaned, standardized, and prepared for use in the model's filtering logic:

  - `Site Type`: For content-based filtering (e.g., ensuring recommendations are relevant to residential vs. commercial properties).
  - `City` and `Site Zip Code`: For geographic-based filtering and ranking.

**Future Feature Engineering (for Advanced Models):**

The following steps are planned for the development of more advanced, predictive models (e.g., using machine learning) but are not required for the current rule-based baseline engine:

- [ ] **Create Temporal Features:** Engineer features based on dates, such as seasonality (e.g., month of application) or trends over time, to capture time-sensitive patterns in rebate adoption.
- [X] **Split Data for Predictive Modeling:** For training and evaluating a machine learning model, the data is now split using a user-based leave-one-out approach. For each user with more than one interaction, the most recent interaction is placed in the test set, and all earlier interactions are placed in the training set. This ensures that every test user is present in the training set, enabling meaningful collaborative filtering evaluation and avoiding cold-start bias in the test set.

### Phase 3: Model Development

- [X] Implement baseline recommendation approaches
  - [X] Popularity-based recommendations
  - [X] Simple content-based filtering
- [X] Develop collaborative filtering model
  - [X] Singular Value Decomposition using Stochastic Gradient Descent for optimization
  - [X] Tune hyperparameters
  - [X] Address cold-start problem
- [X] Develop advanced recommendation model
  - [X] Implement collaborative filtering (SVD)

#### **Algorithm Deep Dive: Singular Value Decomposition (SVD)**

SVD is a powerful matrix factorization technique that serves as a standard for collaborative filtering. It is designed to uncover latent (i.e., hidden) features that connect customers and rebates, allowing for more nuanced recommendations than a purely rule-based system.

**The Core Idea: Finding Hidden Connections**

Imagine a giant table where every row is a unique customer (`APN`) and every column is a rebate program. We place a `1` in a cell if a customer has participated in that rebate and a `0` if they haven't. This is our **user-item interaction matrix**.

The challenge is that this matrix is **sparse**—it's mostly full of zeros because the average customer has only participated in one or two of the six possible rebate programs. It's difficult to see complex patterns just by looking at it.

This is where SVD comes in. It takes our large, sparse user-item matrix and decomposes it into two smaller, much denser matrices:

1. A **Customer-to-Features Matrix**: This matrix describes each customer not by the rebates they have, but by a set of scores for hidden, or **"latent," features**.
2. A **Rebate-to-Features Matrix**: This matrix describes each rebate program using the *exact same set of latent features*.

**What Are "Latent Features"?**

"Latent features" are abstract characteristics that the SVD algorithm discovers on its own from the patterns in the data. They are not concrete things we've defined like `Site Type` or `City`.

In our rebate dataset, a latent feature might learn to represent concepts like:

* **"Major Outdoor Renovation Projects"**: It might notice that customers who get the `Landscape Conversion Rebate` also frequently get the `Irrigation Upgrade Rebate`, so it gives both rebates a high score for this latent feature.
* **"Water-Efficient Fixtures"**: It might find a different pattern connecting other rebates that tend to be adopted together.
* **"Drought-Resistant Gardening"**: Another abstract grouping based on user behavior.

**How SVD Makes a Recommendation**

1. **Learning Profiles:** The SVD algorithm analyzes the training data and creates a profile for every customer and every rebate based on these latent features.
2. **Predicting Interest:** To make a recommendation, the model looks at a customer's profile and compares it to the profiles of all the rebates they *don't* have yet. It calculates a similarity score between the customer and each potential rebate.
3. **Ranking:** A high similarity score means the customer's tastes (as defined by their latent feature scores) align well with the characteristics of the rebate. The model then returns a ranked list of the rebates with the highest scores, as these are the ones the customer is most likely to be interested in.

In short, SVD goes beyond our hand-coded rules to **automatically discover the underlying "taste" profiles of customers and the "characteristic" profiles of rebates**, allowing it to make more nuanced and potentially more accurate recommendations.

#### **Implementation Plan: Advanced Recommendation Model**

**Update (2025-07-22):**

- Data split now uses user-based leave-one-out: for each user with >1 interaction, the most recent is test, all others are train. This ensures SVD/hybrid evaluation is meaningful and fallback is only used for true new users.
- Hybrid model (SVD + EDA boosts) is fully implemented and evaluated. Recommendations are generated by combining SVD collaborative filtering scores with EDA-based re-ranking boosts for city popularity, site type popularity, and rebate correlation. Each recommendation includes both the internal rebate code and display name, as well as a list of human-readable reasons for transparency.
- Fallback (baseline) logic is used only for users with a single historical interaction (true cold-start).
- The feature pipeline now includes Customer Name and Email Address in the cleaned dataset for dashboard and outreach use.
- Evaluation metrics (precision@k, recall@k) are now calculated using this split, and the results show that the hybrid model outperforms both baseline and SVD alone. Baseline is now only used for a small number of users.

This plan outlines the steps to build and evaluate a predictive recommendation model using collaborative filtering to determine if it can provide more accurate recommendations than the established rule-based baseline.

**1. Environment Setup:**

- **[X] Install New Dependency:** Install the `scikit-surprise` library in a dedicated virtual environment (`venv-reco`) running a compatible Python version (3.11) to resolve build issues.

**2. Data Preparation & Splitting (New Notebook Section: `Phase 4.1`):**

- **[X] Create User-Based Leave-One-Out Train/Test Split:**
  - Load the `cleaned_master_data.csv`.
  - For each user with more than one rebate interaction, place their most recent interaction in the test set and all earlier interactions in the training set.
  - This ensures every test user is present in the training set, enabling meaningful collaborative filtering evaluation and avoiding cold-start bias in the test set.
- **[X] Format Data for `surprise`:**
  - Transform the data into the required `(user, item, rating)` format. `Site ID` serves as the user ID, the rebate type as the item ID, and `1` as the implicit rating.

**3. Model Training (New Notebook Section: `Phase 4.2`):**

- **[ ] Load Data into `surprise`:** Load the formatted training data into a `surprise` `Dataset` object.
- **[ ] Train the SVD Model:** Instantiate and train the `SVD` algorithm on the entire training dataset.

**4. Model Evaluation (New Notebook Section: `Phase 4.3`):**

#### 1. Evaluation Strategy for the Baseline Rule-Based Model

Since the baseline model is a direct translation of our EDA findings into a set of logical rules, it cannot be evaluated with traditional accuracy metrics. The evaluation will instead be **qualitative**, focusing on logical correctness and business utility.

**Methodology: Qualitative Review and User Acceptance Testing (UAT)**

The primary goal is to answer the question: **"Are the recommendations relevant, logical, and actionable for the staff?"**

- **[ ] Generate a Diverse Set of Test Cases:** We will create a "golden dataset" of sample customers representing key scenarios (e.g., new vs. existing customers, different site types, different rebate histories) to test the model's logic under various conditions.
- **[ ] Manual Audit of Recommendations:** For each test case, we will manually review the model's output to verify that it correctly applies the rules discovered during EDA. This includes checking for correct filtering, application of bundling/exclusion rules, and relevance of the final ranked list.

**Success Criteria:** The baseline model will be considered a success if it consistently produces recommendations that are **demonstrably logical, explainable, and more useful than having no system at all.** It sets the benchmark that any future model must beat.

#### 2. Evaluation Strategy for Advanced (ML) Model

For the predictive model (hybrid SVD + EDA), we use a rigorous **offline evaluation** based on historical data, with a user-based leave-one-out split. Precision@k and recall@k are calculated for SVD, hybrid, and baseline models. The hybrid model consistently outperforms the baseline and SVD alone, and baseline is now only used for true cold-start users.

**Sample Results (July 2025):**

| Model    | k | Precision@k | Recall@k |
| -------- | - | ----------- | -------- |
| Baseline | 3 | 0.026       | 0.077    |
| Hybrid   | 3 | 0.299       | 0.896    |
| SVD      | 3 | 0.242       | 0.725    |
| Baseline | 5 | 0.022       | 0.078    |
| Hybrid   | 5 | 0.191       | 0.896    |
| SVD      | 5 | 0.191       | 0.896    |

Hybrid and SVD metrics are now meaningful, and fallback is only used for a small number of users.

**Methodology: Offline Evaluation with User-Based Leave-One-Out Split**

- **[X] Create a Hold-Out Test Set:** We split our data using a user-based leave-one-out approach. For each user with more than one rebate interaction, their most recent interaction is placed in the test set, and all earlier interactions are placed in the training set. This ensures every test user is present in the training set, enabling meaningful collaborative filtering evaluation.
- **[X] Calculate Standard Recommendation Metrics:** We compare the model's top *k* recommendations against the actual rebates that users in the test set adopted. Key metrics include:
  - **Precision@k:** Of the top *k* items we recommended, what percentage did the user actually adopt? (Measures accuracy).
  - **Recall@k:** Of the new items the user adopted, what percentage did we successfully recommend in our top *k*? (Measures coverage).

  #### A Practical Example (k=5)

  To make this concrete, let's imagine:
  - The model provides a user their **Top 5 Recommendations**: `[Toilet, Showerhead, Rain Barrel, Sprinkler, Graywater]`
  - In the test period, that same user *actually* adopts **3 new rebates**: `[Showerhead, Sprinkler, Pool Cover]`

  **Calculating Precision@5:**
  - We look at our 5 recommendations and see which ones were correct. The matches are `Showerhead` and `Sprinkler`.
  - **Formula:** (Number of correct recommendations) / k
  - **Result:** `2 / 5 = 40%`
  - *Interpretation: 40% of our recommendations were useful to the user.*

  **Calculating Recall@5:**
  - We look at the 3 rebates the user actually wanted and see how many we found. We found `Showerhead` and `Sprinkler`.
  - **Formula:** (Number of correct recommendations) / (Total number of items the user actually adopted)
  - **Result:** `2 / 3 = 67%`
  - *Interpretation: We successfully identified 67% of the rebates the user was interested in.*

**Success Criteria:** An advanced model is considered successful if it shows a **statistically significant improvement** in `precision@k` and `recall@k` over the established baseline model.

### Project Nuances and Technical Approach

### Files Required for the Project

To build this, we will primarily rely on the two largest and most detailed files, merging them to create a master dataset:

* **`Application List.csv`**: This is our user profile database. It tells us who the customers are and where they are. The key features we'll use from this file are:
  * `Site ID`: The best candidate for a unique "user" identifier.
  * `Site Type`: Crucial for segmenting users (e.g., 'Single Family', 'MFD', 'Commercial').
  * `City`, `Site Zip Code`: The primary features for geographic similarity.
  * `Rebate Status`: Crucial for filtering valid/completed applications.
* **`Appl.Hardware.csv`**: This is our item catalog. It tells us what a user has "consumed" or installed. The key features are:
  * `Rebate Number`: This is the crucial join key to link the hardware back to a user profile in `Application List.csv`.
  * `Hardware`: This is our "item" to be recommended.
  * `Manufacturer`, `Model Name`: These could be used for more granular recommendations.

**`Graywater Rebate.csv`** adds geographic precision with `Latitude` and `Longitude` for more accurate neighborhood definitions.

### Recommendation Engine Design Approach

Our system will use a hybrid approach combining:

#### 1. Behavioral Similarity (Collaborative Filtering)

* The classic "people who installed X also installed Y" logic
* Uses implicit feedback data (installations, not ratings)
* Implemented with Alternating Least Squares (ALS) or matrix factorization with negative sampling

#### 2. Geographic Similarity

* Users in the same area often have similar needs and preferences
* Using `Zip Code` or precise coordinates from graywater data
* Creates neighborhood effects in recommendations

#### 3. Property Similarity

* Segmentation based on `Site Type`
* Ensures recommendations are contextually relevant
* Different recommendation strategies for residential vs. commercial

#### 4. Cold Start Strategy

* Content-based approach for new users
* Based on site characteristics and geographic patterns
* Messaging: "Popular choices for similar properties in your area"

This comprehensive approach ensures relevant recommendations while addressing common challenges in recommendation systems.

---

## SVD Model Implementation: Concepts and Mechanics

This section breaks down the core concepts of the Singular Value Decomposition (SVD) model implemented in this project.

### 1. Model Evaluation Strategy: User-Based Leave-One-Out Splitting

A fundamental challenge in evaluating recommendation models is ensuring the test is fair and simulates a real-world scenario. We achieve this through a **user-based leave-one-out split**.

- **Training Set:** For each user with more than one rebate interaction, contains all but their most recent interaction. The model learns user preferences from this historical data.
- **Test Set:** Contains the most recent interaction for each multi-interaction user.
- **Evaluation Logic:** The model is trained on users' earlier interactions and evaluated on its ability to predict their most recent rebate adoption. We check if the recommendations for a user (based on their training data profile) match the actual most recent rebate they participated in. This approach ensures every test user is present in the training set, enabling meaningful collaborative filtering evaluation and avoiding cold-start bias.

### 2. The "Cold Start" Problem: Handling New Users

A key limitation of SVD is the **"cold start" problem**: the model cannot generate recommendations for new users who have no participation history.

- **SVD Model:** Requires existing user data to build a profile. For a new user, no profile exists.
- **Baseline Model as a Fallback:** Our system uses a hybrid approach. If the SVD model cannot find a user, it defaults to the simpler, rule-based **Baseline Model**. This model provides non-personalized recommendations based on general attributes like city or property type, ensuring every user receives a sensible starting point.

### 3. How SVD Works: From Global Patterns to Personal Profiles

SVD is a powerful technique from linear algebra that uncovers hidden (latent) patterns in the data. The process can be broken into two distinct phases.

#### Phase 1: Training (Slow & Global)

This is the heavy-lifting phase where the model learns from the "collaboration" of all users.

1. **Create the User-Item Matrix:** The model first constructs a huge, sparse matrix where rows are users (`Site ID`) and columns are rebates. A cell has a value of 1 if a user participated in a rebate and 0 otherwise.
2. **Decomposition (The SVD Magic):** The model decomposes this single large matrix into three smaller, dense matrices:
   * **U (User-Factors):** A "User DNA" book. Each row is a user's personal profile, described as a set of scores across several latent features (e.g., a high score for "outdoor-saver" taste, a low score for "indoor-fixer" taste).
   * **V (Item-Factors):** A "Rebate DNA" book. Each column is a rebate's profile, described along the *same* latent features.
   * **Σ (Sigma - Singular Values):** A diagonal matrix that acts as a set of "importance dials." Each value in Sigma corresponds to a latent feature and indicates how important that feature is for explaining the overall behavior of the entire user base.

**The Power of Generalization:** A user's personal profile is not built in isolation. It is determined by how their behavior compares to the generalized patterns (latent features) discovered from the *entire* dataset. This is the "collaborative" aspect of collaborative filtering.

#### Phase 2: Prediction (Fast & Personal)

Once the model is trained, making a recommendation is incredibly efficient.

1. **Identify a User:** The system takes a specific user (e.g., User 309055).
2. **Look Up Profiles:** It looks up that user's profile (a single row in the `U` matrix) and the profile for every rebate they haven't done yet (columns in the `V` matrix).
3. **Calculate a Score:** For each potential rebate, it calculates the dot product of the user's profile and the rebate's profile, weighted by the importance values in Sigma. This produces a single number: the **predicted rating score**.
4. **Rank and Recommend:** The system ranks all the predicted scores from highest to lowest. The final recommendation list is simply the top N items from this ranked list.

## Hybrid Recommendation Model: SVD + EDA Re-ranking

### Motivation

The hybrid recommendation model was developed to address the limitations of pure collaborative filtering (SVD) and rule-based approaches in the context of rebate recommendations. While SVD excels at learning user-item interaction patterns, it can suffer from cold start issues and may overlook important contextual signals such as geography, property type, and domain-specific rebate relationships. The hybrid model enriches SVD predictions with insights from exploratory data analysis (EDA), resulting in more relevant, robust, and explainable recommendations.

### Architecture Overview

The hybrid model operates in two main stages:

1. **Candidate Generation (SVD):**

   - Uses a tuned SVD model from the `surprise` library to generate predicted scores for all rebates a user has not yet participated in.
   - Produces a ranked list of candidate rebates for each user based on collaborative filtering.
2. **Re-ranking with EDA Features:**

   - Each SVD candidate is re-scored using a set of domain-informed "boosts" derived from EDA:
     - **Geographic Popularity Boost:** Rebates that are historically popular in the user’s city receive an additive score.
     - **Site Type Popularity Boost:** Rebates popular for the user’s property/site type receive an additional boost.
     - **Rebate Correlation Boost:** If the user has previously participated in rebates that are strongly positively correlated (e.g., WBIC and Irrigation Upgrade), such candidates are further promoted.
   - The final hybrid score is the sum of the SVD score and all applicable boosts. Candidates are re-ranked by this hybrid score.

### Detailed Logic

- **SVD Scoring:** For each user, the SVD model predicts scores for all rebates not yet participated in, using the anti-testset from the surprise library.
- **Boost Calculations:**
  - **City Popularity:** Calculated as the top-N most common rebates in the user’s city, based on historical data.
  - **Site Type Popularity:** Determined by the top rebates for the user’s property/site type.
  - **Correlation:** Uses the Pearson correlation matrix of rebate participation flags to identify strong positive relationships between rebates.
- **Hybrid Score Formula:**
  - `HybridScore = SVDScore + (CityBoost if in top city rebates) + (SiteTypeBoost if in top site type rebates) + (CorrelationBoost for each strongly correlated rebate user has)`
  - Boost weights are tunable and were optimized via grid search for best precision/recall.

### Evaluation and Performance

- The hybrid model is evaluated alongside baseline and pure SVD models using Precision@k and Recall@k.
- Visualizations and per-user breakdowns are provided for transparency.
- The hybrid approach consistently outperforms the baseline and matches or exceeds SVD in recall, demonstrating its ability to capture both collaborative and contextual signals.

### Practical Implications

- **Improved Relevance:** By incorporating EDA-derived features, the hybrid model delivers recommendations that are more tailored to user context and local trends.
- **Explainability:** The boosting logic makes it easier to explain why a recommendation was made (e.g., "popular in your city", "often paired with rebates you already have").
- **Robustness:** The model mitigates cold start and data sparsity issues by falling back on popularity and correlation when collaborative signals are weak.
- **Flexibility:** Boost weights can be tuned for different business objectives (e.g., prioritizing recall vs. precision).
- **Conflict Resolution:** Negative correlation penalties prevent recommending incompatible rebates, improving recommendation quality and user trust.

### Summary

The hybrid SVD + EDA model represents a best-of-both-worlds approach, leveraging the predictive power of collaborative filtering while grounding recommendations in real-world patterns and domain knowledge. This results in a recommendation engine that is both accurate and actionable for stakeholders and end users.

## Model Interpretability: Latent Feature Analysis

### Motivation for Interpretability

While SVD is powerful for generating predictions, its internal "latent features" are mathematical abstractions that are not directly interpretable. To provide insights into the model's decision-making process and validate its learned patterns, we implemented a comprehensive latent feature analysis using Principal Component Analysis (PCA).

### Technical Implementation

**Data Extraction:**
- Extracted item (rebate) latent factor vectors from the trained SVD model using the `qi` attribute
- Mapped internal sequential IDs back to human-readable rebate names
- Applied PCA with 2 components to reduce dimensionality for visualization

**Analysis Components:**
1. **2D Scatter Plot Visualization:** Shows rebate programs positioned based on their latent feature similarity
2. **Detailed Positioning Table:** Exact Principal Component coordinates for each rebate
3. **Distance Matrix:** Numerical distances between all rebate pairs
4. **Closest Pairs Ranking:** Top 5 most similar rebate combinations

### Key Findings

**Rebate Clustering Patterns:**
- **Irrigation Technology Cluster:** WBIC, Landscape Conversion, and Irrigation Upgrade rebates positioned at nearly identical coordinates (distance ≈ 0.0000)
- **Unique Programs:** Graywater and Rainwater Capture rebates positioned as distinct outliers
- **Moderate Similarity:** Drip Conversion rebate positioned close to but separate from the irrigation cluster

**Business Insights:**
- The model learned that smart irrigation technologies appeal to highly similar user segments
- Graywater and Rainwater Capture programs serve distinct customer niches
- The clustering validates domain knowledge about customer preferences and rebate relationships

### Critical Discovery: SVD/Correlation Contradiction

**The Problem:**
During analysis, we discovered a significant contradiction between the SVD latent features and the Pearson correlation matrix:
- **Correlation Matrix:** Showed strong negative correlation (-0.63) between Landscape Conversion and other irrigation rebates
- **SVD Clustering:** Positioned these same rebates at identical coordinates, suggesting perfect similarity

**Root Cause Analysis:**
The contradiction was caused by **extreme data sparsity** in irrigation rebate adoptions:
- Very few users adopted multiple irrigation rebates
- The SVD model couldn't distinguish between rarely co-occurring items
- This resulted in identical latent feature vectors for sparse rebates

**Resolution: Enhanced Hybrid Model Logic**
To resolve this contradiction, we implemented a **negative correlation penalty** in the hybrid model:

```python
# Enhanced correlation logic
correlation = correlation_matrix.loc[existing_rebate, item]
if correlation > 0.5:
    hybrid_score += boost_weight  # Positive correlation boost
elif correlation < -0.3:
    hybrid_score -= penalty_weight  # Negative correlation penalty
```

**Impact:**
- Users with WBIC rebates now receive active penalties for Landscape Conversion recommendations
- Conflicting rebates are discouraged with transparent explanations
- The hybrid model now trusts correlation patterns over flawed SVD clustering for sparse items

### Validation and Business Value

**Model Reliability:**
- The latent feature analysis revealed both strengths and limitations of the SVD approach
- Identified cases where domain knowledge (correlation matrix) is more reliable than collaborative filtering
- Enhanced the hybrid model to handle edge cases and sparse data scenarios

**Stakeholder Benefits:**
- **Transparency:** Staff can understand why certain rebates cluster together
- **Conflict Prevention:** System actively avoids recommending incompatible rebates
- **Confidence:** Clear evidence that the model learns meaningful patterns from customer behavior

**Technical Lessons:**
- Collaborative filtering excels with dense interaction data but struggles with sparse items
- Hybrid approaches can compensate for individual model limitations
- Interpretability analysis is crucial for identifying and resolving model contradictions

- [x] Develop collaborative filtering model
- [x] Implement latent feature interpretability analysis
- [x] Resolve SVD/correlation contradiction with negative penalty logic

---

## Deployment Phases

### **Phase 1: Model Packaging (Next Step)**
- Save the trained SVD model as .pkl file
- Package all supporting data (correlation matrix, city/site type popularity)
- Create model loading functions

### **Phase 2: API Development**
- Build FastAPI application with recommendation endpoints
- Integrate the packaged model into the API
- This is when the model starts serving recommendations

### **Phase 3: Bulk Recommendation Generation (Optional)**
- Run the model against ALL customers in cleaned_master_data.csv
- Pre-generate recommendations for faster dashboard loading
- Store results for immediate dashboard consumption
