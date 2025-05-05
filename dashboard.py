import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def load_data_ap():
    df = pd.read_csv("AP001.csv", parse_dates=["From Date", "To Date"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

@st.cache_data
def load_data_city():
    df = pd.read_csv("city_day.csv", parse_dates=["Date"])
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df = df[df['City'] != "Ahmedabad"]
    df = df[df['City'] != "Guwahati"]
    
    return df

@st.cache_data
def load_data_temp():
    df = pd.read_csv(
        "GlobalLandTemperaturesByMajorCity.csv",
        parse_dates=["dt"]
    )
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def main():
    st.set_page_config(page_title="Ordin 25 Dashboard", layout="wide")
    st.title("Ordin 25 Dashboard")
    st.sidebar.markdown(
        "<h1>Configuration</h1>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("**Select Dataset**")
    dataset = st.sidebar.selectbox(
        "",
        ["National AQI Trends", "City AQI Trends", "Global Temperature"]
    )
    st.sidebar.markdown("---")

    if dataset == "National AQI Trends":
        df = load_data_ap()
        st.header("National AQI Trends Dataset Overview")
        st.write(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe())

        pollutants = st.multiselect(
            "Select pollutants to plot",
            options=df.columns[2:],
            default=["PM2.5 (ug/m3)", "PM10 (ug/m3)"]
        )
        date_range = st.date_input("Select date range",[df["From Date"].min(), df["From Date"].max()])
        mask = (            (df["From Date"] >= pd.to_datetime(date_range[0])) &
            (df["From Date"] <= pd.to_datetime(date_range[1]))
        )
        df_filtered = df.loc[mask]

        if pollutants:
            fig = px.line(
                df_filtered,
                x="From Date",
                y=pollutants,
                title="Pollutant Time Series"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df[pollutants].corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2, cbar=True)
        st.pyplot(fig2)
        st.subheader("Missing Data Summary")
        st.write(df_filtered.isnull().sum())


        st.subheader("Pollutant Distribution")
        for p in pollutants:
            fig_hist = px.histogram(
                df_filtered, x=p, nbins=30, title=f"Distribution of {p}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)


        st.subheader("Monthly Average Trends")
        df_month = (
            df_filtered
            .resample('M', on="From Date")[pollutants]
            .mean()
            .reset_index()
        )
        fig_month = px.line(
            df_month, x="From Date", y=pollutants, title="Monthly Averages"
        )
        st.plotly_chart(fig_month, use_container_width=True)

    elif dataset == "City AQI Trends":
        df = load_data_city()
        st.header("City AQI Trends Overview")
        st.write(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe())

        cities = df["City"].unique().tolist()
        selected_cities = st.multiselect(
            "Select cities",
            options=cities,
            default=cities[:15]
        )
        aqi_metric = st.selectbox(
            "Select AQI metric",
            options=["AQI", "PM2.5", "PM10", "O3", "NO2"]
        )

        if selected_cities:
            df_cities = df[df["City"].isin(selected_cities)]
            fig3 = px.line(
                df_cities,
                x="Date",
                y=aqi_metric,
                color="City",
                title=f"{aqi_metric} Trends by City")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("AQI Bucket Distribution")
        fig4, ax4 = plt.subplots()
        sns.countplot(
            data=df,
            x="AQI_Bucket",
            order=df["AQI_Bucket"].value_counts().index,
            ax=ax4
        )
        ax4.set_title("AQI Bucket Counts")
        st.pyplot(fig4)

        st.subheader("AQI Violin Plot by City")
        fig5, ax5 = plt.subplots(figsize=(15, 6))
        sns.violinplot(
            data=df[df["City"].isin(selected_cities)],
            x="City",
            y="AQI",
            ax=ax5
        )
        st.pyplot(fig5)


        st.subheader("Average AQI by City")
        df_avg = (
            df[df["City"].isin(selected_cities)]
            .groupby("City")[aqi_metric]
            .mean()
            .reset_index()
        )
        fig_avg = px.bar(
            df_avg, x="City", y=aqi_metric, title=f"Avg {aqi_metric} by City"
        )
        st.plotly_chart(fig_avg, use_container_width=True)


        st.subheader("Overall Monthly AQI Trends")
        df_city_month = (df
            .groupby(pd.Grouper(key="Date", freq="M"))[aqi_metric]
            .mean()
            .reset_index()
        )
        fig_city_month = px.line(
            df_city_month, x="Date", y=aqi_metric, title="Monthly AQI Trends"
        )
        st.plotly_chart(fig_city_month, use_container_width=True)

    else:
        df_temp = load_data_temp()
        st.header("Global Land Temperatures by Major City")
        st.write(df_temp.head())
        st.subheader("Summary Statistics")
        st.write(df_temp.describe())


        cities = df_temp["City"].unique().tolist()
        selected_cities = st.multiselect(
            "Select cities",
            options=cities,
            default=cities[:10]
        )
        date_range = st.date_input(
            "Select date range",
            [df_temp["dt"].min(), df_temp["dt"].max()]
        )
        mask = (
            (df_temp["dt"] >= pd.to_datetime(date_range[0])) &
            (df_temp["dt"] <= pd.to_datetime(date_range[1]))
        )
        df_temp_filt = df_temp.loc[mask]


        st.subheader("Temperature Trends by City")
        fig_temp = px.line(
            df_temp_filt[df_temp_filt["City"].isin(selected_cities)],
            x="dt", y="AverageTemperature", color="City",
            title="Temperature Trends"
        )
        st.plotly_chart(fig_temp, use_container_width=True)


        st.subheader("Global Temperature Distribution")
        fig_hist_temp = px.histogram(
            df_temp_filt, x="AverageTemperature",
            nbins=50, title="Temperature Distribution"
        )
        st.plotly_chart(fig_hist_temp, use_container_width=True)


        st.subheader("Average Temperature by Country")
        df_country = (
            df_temp_filt.groupby("Country")["AverageTemperature"]
            .mean().reset_index()
        )
        fig_bar_country = px.bar(
            df_country.sort_values("AverageTemperature", ascending=False),
            x="Country", y="AverageTemperature",
            title="Avg Temp by Country"
        )
        st.plotly_chart(fig_bar_country, use_container_width=True)


        # st.subheader("Latest Global Temperatures Map")
        # latest = df_temp_filt["dt"].max()
        # df_latest = df_temp_filt[
        #     (df_temp_filt["dt"] == latest) &
        #     (df_temp_filt["AverageTemperature"].notna())
        # ]
        # def parse_coord(x):
        #     if pd.isna(x): return None
        #     direction = x[-1]
        #     val = float(x[:-1])
        #     return val if direction in ("N","E") else -val
        # df_latest["lat"] = df_latest["Latitude"].apply(parse_coord)
        # df_latest["lon"] = df_latest["Longitude"].apply(parse_coord)
        # df_latest = df_latest.dropna(subset=["lat","lon"])

        # fig_map = px.scatter_geo(
        #     df_latest,
        #     lat="lat", lon="lon",
        #     color="AverageTemperature",
        #     hover_name="City",
        #     size="AverageTemperature",
        #     projection="natural earth",
        #     title=f"Global Temps on {latest.date()}"
        # )
        # st.plotly_chart(fig_map, use_container_width=True)
        
if __name__ == "__main__":
    main()