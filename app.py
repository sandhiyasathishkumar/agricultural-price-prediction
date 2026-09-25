import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agricultural Commodity Price Prediction",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

file_path = "Agricultural_Price_Prediction.csv.csv"

df = pd.read_csv(file_path)


# ============================================================
# TITLE
# ============================================================

st.title("🌾 Agricultural Commodity Price Prediction Dashboard")

st.write(
    "This dashboard presents exploratory data analysis, "
    "machine learning model performance, and commodity price "
    "search using agricultural market data."
)

st.divider()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Dataset Records",
    len(df)
)

col2.metric(
    "Number of Features",
    len(df.columns)
)

col3.metric(
    "Commodities",
    df["Commodity"].nunique()
)

col4.metric(
    "States",
    df["State"].nunique()
)


# ============================================================
# PRICE SUMMARY
# ============================================================

st.header("💰 Price Summary")

price_col1, price_col2, price_col3 = st.columns(3)

price_col1.metric(
    "Average Minimum Price",
    f"₹{df['Min_x0020_Price'].mean():,.2f}"
)

price_col2.metric(
    "Average Maximum Price",
    f"₹{df['Max_x0020_Price'].mean():,.2f}"
)

price_col3.metric(
    "Average Modal Price",
    f"₹{df['Modal_x0020_Price'].mean():,.2f}"
)


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

st.header("📈 Exploratory Data Analysis")

st.write(
    "The following visualizations show the distribution of prices, "
    "commodity frequency, state-wise patterns, grade-wise patterns, "
    "district-wise patterns, and relationships among price variables."
)


# ============================================================
# PLOT 1 - MODAL PRICE HISTOGRAM
# ============================================================

st.subheader("1. Distribution of Agricultural Commodity Modal Prices")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    df["Modal_x0020_Price"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Modal Price")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Agricultural Commodity Modal Prices")

st.pyplot(fig)

st.info(
    "Insight: The histogram shows that most modal price values are "
    "concentrated in the lower price range, while a smaller number "
    "of observations have very high modal prices.\n\n"
    "Inference: This indicates that lower-priced commodities are more "
    "frequently represented in the dataset, while high-priced "
    "observations occur less frequently. The distribution is right-skewed "
    "due to the presence of high-price observations."
)


# ============================================================
# PLOT 2 - MINIMUM PRICE HISTOGRAM
# ============================================================

st.subheader("2. Distribution of Agricultural Commodity Minimum Prices")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    df["Min_x0020_Price"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Minimum Price")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Agricultural Commodity Minimum Prices")

st.pyplot(fig)

st.info(
    "Insight: The minimum prices are mainly concentrated in the lower "
    "price range, with fewer observations having very high values.\n\n"
    "Inference: The distribution indicates that most commodities have "
    "relatively lower minimum market prices."
)


# ============================================================
# PLOT 3 - MAXIMUM PRICE HISTOGRAM
# ============================================================

st.subheader("3. Distribution of Agricultural Commodity Maximum Prices")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    df["Max_x0020_Price"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Maximum Price")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Agricultural Commodity Maximum Prices")

st.pyplot(fig)

st.info(
    "Insight: Most maximum price observations occur in the lower "
    "price range, while a smaller number of observations have very "
    "high maximum prices.\n\n"
    "Inference: The presence of high-price observations creates a "
    "right-skewed distribution."
)


# ============================================================
# PLOT 4 - TOP 10 COMMODITIES
# ============================================================

st.subheader("4. Top 10 Most Frequent Agricultural Commodities")

commodity_counts = df["Commodity"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    commodity_counts.index.astype(str),
    commodity_counts.values,
    edgecolor="black"
)

ax.set_xlabel("Commodity")
ax.set_ylabel("Number of Records")
ax.set_title("Top 10 Most Frequent Agricultural Commodities")

plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The bar chart shows the top 10 most frequently recorded "
    "commodities in the dataset.\n\n"
    "Inference: Some commodities have greater representation than "
    "others, which reflects differences in the available market records."
)


# ============================================================
# PLOT 5 - AVERAGE MODAL PRICE OF TOP 10 COMMODITIES
# ============================================================

st.subheader("5. Average Modal Price of Top 10 Commodities")

top_commodities = df["Commodity"].value_counts().head(10).index

commodity_price = (
    df[df["Commodity"].isin(top_commodities)]
    .groupby("Commodity")["Modal_x0020_Price"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    commodity_price.index.astype(str),
    commodity_price.values,
    edgecolor="black"
)

ax.set_xlabel("Commodity")
ax.set_ylabel("Average Modal Price")
ax.set_title("Average Modal Price of Top 10 Commodities")

plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The average modal price differs among the top ten "
    "frequently recorded commodities.\n\n"
    "Inference: Commodity type is associated with differences in "
    "the observed modal price levels."
)


# ============================================================
# PLOT 6 - AVERAGE MODAL PRICE BY STATE
# ============================================================

st.subheader("6. Average Modal Price by State")

state_price = (
    df.groupby("State")["Modal_x0020_Price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    state_price.index.astype(str),
    state_price.values,
    edgecolor="black"
)

ax.set_xlabel("State")
ax.set_ylabel("Average Modal Price")
ax.set_title("Average Modal Price by State")

plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The average modal price varies across states.\n\n"
    "Inference: Regional market conditions are associated with "
    "differences in agricultural commodity prices."
)


# ============================================================
# PLOT 7 - MODAL PRICE DISTRIBUTION BY TOP 10 COMMODITIES
# ============================================================

st.subheader("7. Modal Price Distribution by Top 10 Commodities")

plot_data = df[
    df["Commodity"].isin(top_commodities)
]

fig, ax = plt.subplots(figsize=(12, 6))

plot_data.boxplot(
    column="Modal_x0020_Price",
    by="Commodity",
    ax=ax,
    grid=False
)

ax.set_title("Modal Price Distribution by Top 10 Commodities")
ax.set_xlabel("Commodity")
ax.set_ylabel("Modal Price")

plt.suptitle("")
plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The box plot shows differences in the median, spread, "
    "and outliers of modal prices among the top commodities.\n\n"
    "Inference: Different commodities have different modal price "
    "levels and price variability."
)


# ============================================================
# PLOT 8 - MINIMUM PRICE DISTRIBUTION BY GRADE
# ============================================================

st.subheader("8. Minimum Price Distribution by Grade")

fig, ax = plt.subplots(figsize=(8, 5))

df.boxplot(
    column="Min_x0020_Price",
    by="Grade",
    ax=ax,
    grid=False
)

ax.set_title("Minimum Price Distribution by Grade")
ax.set_xlabel("Grade")
ax.set_ylabel("Minimum Price")

plt.suptitle("")

st.pyplot(fig)

st.info(
    "Insight: Minimum price distributions vary across the different "
    "grades in the dataset.\n\n"
    "Inference: Grade is associated with differences in observed "
    "minimum agricultural commodity prices."
)


# ============================================================
# PLOT 9 - MAXIMUM PRICE DISTRIBUTION BY GRADE
# ============================================================

st.subheader("9. Maximum Price Distribution by Grade")

fig, ax = plt.subplots(figsize=(8, 5))

df.boxplot(
    column="Max_x0020_Price",
    by="Grade",
    ax=ax,
    grid=False
)

ax.set_title("Maximum Price Distribution by Grade")
ax.set_xlabel("Grade")
ax.set_ylabel("Maximum Price")

plt.suptitle("")

st.pyplot(fig)

st.info(
    "Insight: The maximum price distribution differs among grades.\n\n"
    "Inference: The grade classification is related to variations "
    "in the maximum prices observed in the market data."
)


# ============================================================
# PLOT 10 - MINIMUM PRICE VS MODAL PRICE
# ============================================================

st.subheader("10. Relationship Between Minimum Price and Modal Price")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Min_x0020_Price"],
    df["Modal_x0020_Price"],
    alpha=0.5
)

ax.set_xlabel("Minimum Price")
ax.set_ylabel("Modal Price")
ax.set_title("Minimum Price vs Modal Price")

st.pyplot(fig)

st.info(
    "Insight: The scatter plot shows an overall positive relationship "
    "between minimum price and modal price.\n\n"
    "Inference: Higher minimum prices are generally associated with "
    "higher modal prices."
)


# ============================================================
# PLOT 11 - MAXIMUM PRICE VS MODAL PRICE
# ============================================================

st.subheader("11. Relationship Between Maximum Price and Modal Price")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Max_x0020_Price"],
    df["Modal_x0020_Price"],
    alpha=0.5
)

ax.set_xlabel("Maximum Price")
ax.set_ylabel("Modal Price")
ax.set_title("Modal Price vs Maximum Price")

st.pyplot(fig)

st.info(
    "Insight: The scatter plot shows an overall upward pattern "
    "between maximum price and modal price.\n\n"
    "Inference: Maximum Price provides useful information for "
    "understanding variations in Modal Price."
)


# ============================================================
# PLOT 12 - MINIMUM PRICE VS MAXIMUM PRICE
# ============================================================

st.subheader("12. Relationship Between Minimum Price and Maximum Price")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Min_x0020_Price"],
    df["Max_x0020_Price"],
    alpha=0.5
)

ax.set_xlabel("Minimum Price")
ax.set_ylabel("Maximum Price")
ax.set_title("Minimum Price vs Maximum Price")

st.pyplot(fig)

st.info(
    "Insight: Minimum and maximum prices show a positive relationship "
    "across the observations.\n\n"
    "Inference: Market observations with higher minimum prices generally "
    "also have higher maximum prices."
)


# ============================================================
# PLOT 13 - CORRELATION HEATMAP
# ============================================================

st.subheader("13. Correlation Heatmap of Price Variables")

correlation = df[
    [
        "Min_x0020_Price",
        "Max_x0020_Price",
        "Modal_x0020_Price"
    ]
].corr()

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

ax.set_xticks(range(len(correlation.columns)))
ax.set_yticks(range(len(correlation.columns)))

ax.set_xticklabels(correlation.columns)
ax.set_yticklabels(correlation.columns)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        ax.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

ax.set_title("Correlation Heatmap of Price Variables")

fig.colorbar(im)

st.pyplot(fig)

st.info(
    "Insight: The heatmap shows the correlation between minimum, "
    "maximum, and modal prices.\n\n"
    "Inference: The price variables have strong positive relationships "
    "with each other, indicating that they move together across many "
    "market observations."
)


# ============================================================
# PLOT 14 - AVERAGE MINIMUM, MAXIMUM AND MODAL PRICES
# ============================================================

st.subheader("14. Average Minimum, Maximum and Modal Prices")

average_prices = [
    df["Min_x0020_Price"].mean(),
    df["Max_x0020_Price"].mean(),
    df["Modal_x0020_Price"].mean()
]

price_names = [
    "Minimum Price",
    "Maximum Price",
    "Modal Price"
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    price_names,
    average_prices,
    edgecolor="black"
)

ax.set_xlabel("Price Type")
ax.set_ylabel("Average Price (₹)")
ax.set_title("Average Minimum, Maximum and Modal Prices")

st.pyplot(fig)

st.info(
    "Insight: The bar chart compares the average minimum, maximum, "
    "and modal prices in the dataset.\n\n"
    "Inference: The three price measures have different average values, "
    "with the maximum price having a higher average than the minimum "
    "and modal prices."
)


# ============================================================
# PLOT 15 - AVERAGE MODAL PRICE BY GRADE
# ============================================================

st.subheader("15. Average Modal Price by Grade")

grade_price = (
    df.groupby("Grade")["Modal_x0020_Price"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    grade_price.index.astype(str),
    grade_price.values,
    edgecolor="black"
)

ax.set_xlabel("Grade")
ax.set_ylabel("Average Modal Price (₹)")
ax.set_title("Average Modal Price by Grade")

st.pyplot(fig)

st.info(
    "Insight: The bar chart shows the average modal price for each "
    "grade available in the dataset.\n\n"
    "Inference: The average modal price varies across grades, indicating "
    "that grade is associated with differences in commodity price levels."
)


# ============================================================
# PLOT 16 - AVERAGE MODAL PRICE BY TOP 10 DISTRICTS
# ============================================================

st.subheader("16. Average Modal Price by Top 10 Districts")

top_districts = df["District"].value_counts().head(10).index

district_price = (
    df[df["District"].isin(top_districts)]
    .groupby("District")["Modal_x0020_Price"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    district_price.index.astype(str),
    district_price.values,
    edgecolor="black"
)

ax.set_xlabel("District")
ax.set_ylabel("Average Modal Price (₹)")
ax.set_title("Average Modal Price by Top 10 Districts")

plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The chart compares the average modal prices of the top "
    "10 districts based on the number of records.\n\n"
    "Inference: Modal price levels vary across districts, indicating "
    "regional differences in agricultural commodity prices."
)


# ============================================================
# PLOT 17 - MODAL PRICE DISTRIBUTION BY TOP 10 COMMODITIES
# ============================================================

st.subheader("17. Modal Price Distribution by Top 10 Commodities")

top_commodities_plot = (
    df["Commodity"]
    .value_counts()
    .head(10)
    .index
)

plot_data_17 = df[
    df["Commodity"].isin(top_commodities_plot)
]

fig, ax = plt.subplots(figsize=(12, 6))

plot_data_17.boxplot(
    column="Modal_x0020_Price",
    by="Commodity",
    ax=ax,
    grid=False
)

ax.set_title("Modal Price Distribution by Top 10 Commodities")
ax.set_xlabel("Commodity")
ax.set_ylabel("Modal Price (₹)")

plt.suptitle("")
plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The box plot shows the distribution of modal prices for "
    "the top 10 most frequently recorded commodities. The median, "
    "spread, and outliers vary across the commodities.\n\n"
    "Inference: This indicates that different commodities have different "
    "modal price levels and price variability. The presence of outliers "
    "shows that some commodities have occasional observations with "
    "unusually high modal prices."
)


# ============================================================
# PLOT 18 - MODAL PRICE VS MAXIMUM PRICE
# ============================================================

st.subheader("18. Modal Price vs Maximum Price")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["Max_x0020_Price"],
    df["Modal_x0020_Price"],
    edgecolor="black",
    alpha=0.6
)

ax.set_title("Modal Price vs Maximum Price")
ax.set_xlabel("Maximum Price (₹)")
ax.set_ylabel("Modal Price (₹)")

plt.tight_layout()

st.pyplot(fig)

st.info(
    "Insight: The scatter plot shows the relationship between maximum "
    "price and modal price. The points show an overall upward pattern, "
    "indicating that higher maximum prices are generally associated "
    "with higher modal prices.\n\n"
    "Inference: This indicates a positive relationship between maximum "
    "price and modal price. Maximum Price therefore provides useful "
    "information for understanding variations in the Modal Price."
)


# ============================================================
# PLOT 19 - NUMBER OF RECORDS BY GRADE
# ============================================================

st.subheader("19. Number of Records by Grade")

grade_counts = df["Grade"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    grade_counts.index.astype(str),
    grade_counts.values,
    edgecolor="black"
)

ax.set_title("Number of Records by Grade")
ax.set_xlabel("Grade")
ax.set_ylabel("Number of Records")

st.pyplot(fig)

st.info(
    "Insight: The bar chart shows the number of records available for "
    "each grade in the dataset. The number of observations varies "
    "across the different grades.\n\n"
    "Inference: This indicates that the dataset does not contain an "
    "equal number of records for every grade. The distribution of "
    "records across grades should be considered when analysing the "
    "effect of Grade on agricultural commodity prices."
)


# ============================================================
# PLOT 20 - TOP 10 STATES BY NUMBER OF RECORDS
# ============================================================

st.subheader("20. Top 10 States by Number of Records")

top_states = df["State"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    top_states.index.astype(str),
    top_states.values,
    edgecolor="black"
)

ax.set_title("Top 10 States by Number of Records")
ax.set_xlabel("State")
ax.set_ylabel("Number of Records")

plt.xticks(rotation=45)

st.pyplot(fig)

st.info(
    "Insight: The bar chart shows the top 10 states based on the "
    "number of records available in the dataset. The number of "
    "observations varies across the states.\n\n"
    "Inference: This indicates that some states have a higher "
    "representation in the dataset than others. The difference in "
    "record counts should be considered when analysing regional "
    "patterns in agricultural commodity prices."
)


# ============================================================
# MACHINE LEARNING MODEL PERFORMANCE
# ============================================================

st.header("🤖 Machine Learning Model Performance")

model_results = pd.DataFrame({
    "Model": [
        "Multiple Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree Regression"
    ],
    "MAE": [
        1270.3851,
        1327.2000,
        1268.0500,
        2063.8093
    ],
    "RMSE": [
        2932.5473,
        2766.3600,
        3070.2200,
        3788.3532
    ],
    "R² Score": [
        0.7220,
        0.7526,
        0.6953,
        0.5360
    ]
})

st.dataframe(
    model_results,
    use_container_width=True
)


# ============================================================
# R2 COMPARISON CHART
# ============================================================

st.subheader("R² Score Comparison")

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    model_results["Model"],
    model_results["R² Score"],
    edgecolor="black"
)

ax.set_xlabel("Model")
ax.set_ylabel("R² Score")
ax.set_title("R² Score Comparison of Machine Learning Models")

plt.xticks(rotation=20)

st.pyplot(fig)

st.info(
    "Insight: The chart compares the R² scores of the four regression "
    "models used in the project.\n\n"
    "Inference: The R² values show how well the models explain the "
    "variation in agricultural commodity modal prices."
)


# ============================================================
# HYPERPARAMETER EVALUATION
# ============================================================

st.header("⚙️ Hyperparameter Evaluation")


# ============================================================
# RIDGE HYPERPARAMETERS
# ============================================================

st.subheader("Ridge Regression - Alpha Values")

ridge_results = pd.DataFrame({
    "Alpha": [0.01, 0.1, 1, 10, 100],
    "MAE": [
        1276.06,
        1283.15,
        1327.20,
        1384.61,
        1849.57
    ],
    "RMSE": [
        2915.63,
        2806.53,
        2766.36,
        3285.43,
        4182.60
    ],
    "R² Score": [
        0.7252,
        0.7454,
        0.7526,
        0.6511,
        0.4345
    ]
})

st.dataframe(
    ridge_results,
    use_container_width=True
)

st.info(
    "Insight: Ridge Regression was evaluated using different alpha "
    "values to study the effect of regularization.\n\n"
    "Inference: The performance changes with the regularization "
    "strength, showing that the choice of alpha affects model performance."
)


# ============================================================
# LASSO HYPERPARAMETERS
# ============================================================

st.subheader("Lasso Regression - Alpha Values")

lasso_results = pd.DataFrame({
    "Alpha": [0.01, 0.1, 1, 10, 100],
    "MAE": [
        1283.08,
        1277.41,
        1268.05,
        1528.60,
        2672.17
    ],
    "RMSE": [
        3094.36,
        3099.97,
        3070.22,
        3642.29,
        5073.82
    ],
    "R² Score": [
        0.6905,
        0.6893,
        0.6953,
        0.5711,
        0.1678
    ]
})

st.dataframe(
    lasso_results,
    use_container_width=True
)

st.info(
    "Insight: Lasso Regression was evaluated using different alpha "
    "values to study the effect of regularization.\n\n"
    "Inference: The model performance varies with alpha, indicating "
    "that regularization strength influences prediction performance."
)


# ============================================================
# DECISION TREE HYPERPARAMETERS
# ============================================================

st.subheader("Decision Tree - Maximum Depth")

tree_results = pd.DataFrame({
    "Maximum Depth": [3, 5, 7, 10, 15],
    "MAE": [
        2762.51,
        2405.97,
        2293.85,
        2198.48,
        2069.26
    ],
    "RMSE": [
        5297.14,
        4873.87,
        4316.66,
        3933.44,
        3813.21
    ],
    "R² Score": [
        0.0929,
        0.2321,
        0.3976,
        0.4998,
        0.5299
    ]
})

st.dataframe(
    tree_results,
    use_container_width=True
)

st.info(
    "Insight: The Decision Tree was evaluated using different maximum "
    "depth values.\n\n"
    "Inference: The evaluation shows that changing the tree depth "
    "affects the prediction performance of the model."
)


# ============================================================
# RAG / COMMODITY PRICE SEARCH
# ============================================================

st.header("🔎 Commodity Price Search")

st.write(
    "Search the agricultural market dataset to retrieve price "
    "information for a selected commodity."
)

commodity_list = sorted(
    df["Commodity"].dropna().unique()
)

selected_commodity = st.selectbox(
    "Select a Commodity",
    commodity_list
)

search_results = df[
    df["Commodity"].str.lower()
    == selected_commodity.lower()
]

search_columns = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Arrival_Date",
    "Modal_x0020_Price"
]

st.dataframe(
    search_results[search_columns].head(10),
    use_container_width=True
)


# ============================================================
# RAG-STYLE PRICE CHATBOT
# ============================================================

st.header("💬 Agricultural Price Query Chatbot")

st.write(
    "Ask a simple question such as: "
    "`What is the price of onion?`"
)

user_query = st.text_input(
    "Enter your price query:"
)


def search_price(query):

    query = query.lower().strip()

    matched_data = df[
        df["Commodity"]
        .str.lower()
        .str.contains(query, na=False)
    ]

    return matched_data


if user_query:

    # Remove common words from a natural-language query
    query_words = user_query.lower().split()

    stop_words = {
        "what",
        "is",
        "the",
        "price",
        "of",
        "for",
        "commodity",
        "market",
        "today",
        "give",
        "me",
        "tell",
        "show",
        "please"
    }

    commodity_words = [
        word.strip("?,.")
        for word in query_words
        if word.strip("?,.") not in stop_words
    ]

    matched_data = pd.DataFrame()

    # Try searching using the remaining commodity words
    for word in commodity_words:

        temp_data = search_price(word)

        if len(temp_data) > 0:
            matched_data = temp_data
            break

    if len(matched_data) > 0:

        st.success(
            f"Found {len(matched_data)} matching records."
        )

        chatbot_columns = [
            "State",
            "District",
            "Market",
            "Commodity",
            "Variety",
            "Grade",
            "Arrival_Date",
            "Modal_x0020_Price"
        ]

        st.dataframe(
            matched_data[chatbot_columns].head(10),
            use_container_width=True
        )

        average_price = matched_data[
            "Modal_x0020_Price"
        ].mean()

        st.write(
            f"**Average Modal Price:** ₹{average_price:,.2f}"
        )

    else:

        st.warning(
            "No matching commodity was found. "
            "Please enter a commodity name such as Onion, Potato, "
            "Tomato, Brinjal or Bhindi."
        )


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

st.header("📌 Dashboard Summary")

st.write(
    f"""
    - **Dataset Records:** {len(df)}
    - **Number of Features:** {len(df.columns)}
    - **Number of Commodities:** {df["Commodity"].nunique()}
    - **Average Minimum Price:** ₹{df["Min_x0020_Price"].mean():,.2f}
    - **Average Maximum Price:** ₹{df["Max_x0020_Price"].mean():,.2f}
    - **Average Modal Price:** ₹{df["Modal_x0020_Price"].mean():,.2f}
    - **Commodity Search:** Available
    - **Price Query Feature:** Available
    """
)


# ============================================================
# FINAL RESULT
# ============================================================

st.header("✅ Result")

st.write(
    "The agricultural commodity price prediction system was "
    "successfully developed using machine learning techniques. "
    "Exploratory Data Analysis was performed to understand "
    "commodity prices, states, districts, grades, and relationships "
    "among price variables. Multiple regression models and Decision "
    "Tree Regression were evaluated using performance metrics and "
    "hyperparameter variations. A RAG-style price query system was "
    "also developed to retrieve commodity price information from "
    "the agricultural market dataset."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agricultural Commodity Price Prediction | "
    "Machine Learning Project"
)
