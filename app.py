import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandasdmx as pdmx
import pandas_datareader.data as web


# Plot settings
plt.style.use(
    "https://github.com/aeturrell/coding-for-economists/raw/main/plot_style.txt"
)

# Page settings
st.set_page_config(
    page_title="An example streamlit app",
    page_icon="👩‍💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# This section is just data retrieval and not directly about the dashboard
# ----------------------------------------------------------------
# settings to retrieve OECD data
industry_dict = {
    "B1GVA": "Agriculture, forestry, and fishing.",
    "B1GVB_E": "Industry, including energy",
    "B1GVF": "Construction",
    "B1GVG_I": "Distrib. trade",
    "B1GVJ": "Information and communication",
    "B1GVK": "Financial and insurance",
    "B1GVL": "Real estate",
    "B1GVM_N": "Prof. services",
    "B1GVO_Q": "Public Admin",
    "B1GVR_U": "Other services",
}
codes = list(industry_dict.keys())
industry_names = list(industry_dict.values())

# Functions
@st.cache
def get_oecd_data():
    """Grabs OECD data in tidy format.

    :return: OECD output data by time-sector for UK
    :rtype: pandas dataframe
    """
    # Tell pdmx we want OECD data
    oecd = pdmx.Request("OECD")
    # Set out everything about the request in the format specified by the OECD API
    data = oecd.data(
        resource_id="QNA",
        key="GBR." + "+".join(codes) + ".LNBQRSA.Q/all?startTime=2015",
    ).to_pandas()

    df_oecd = pd.DataFrame(data).reset_index()
    # some clean up operations
    df_oecd["datetime"] = (
        pd.to_datetime(
            df_oecd["TIME_PERIOD"].apply(
                lambda x: str(x[:4]) + "-" + str(int(x[-1]) * 3)
            ),
            format="%Y-%m",
        )
        + pd.offsets.MonthEnd()
    )
    df_oecd["value"] = df_oecd["value"]
    df_oecd["industry"] = df_oecd["SUBJECT"].map(industry_dict)
    df_oecd = df_oecd.sort_values(["industry", "datetime"])
    # find the greatest growing sector within each time-sector cell
    # first compute growth within industry
    df_oecd["growth"] = df_oecd.groupby(["industry"])["value"].transform(
        lambda x: x.pct_change()
    )
    # now max growth within each time-period
    df_oecd["max"] = df_oecd.groupby("datetime")["growth"].transform(lambda x: x.max())
    df_oecd["argmax"] = (
        df_oecd.groupby(["datetime"])["growth"]
        .transform(lambda x: x.idxmax())
        .astype("Int32")
    )
    df_oecd["max_ind"] = pd.NA
    df_oecd.loc[:, "max_ind"] = df_oecd.loc[
        df_oecd["argmax"].fillna(0), "industry"
    ].values
    # and min
    df_oecd["min"] = df_oecd.groupby("datetime")["growth"].transform(lambda x: x.min())
    df_oecd["argmin"] = (
        df_oecd.groupby(["datetime"])["growth"]
        .transform(lambda x: x.idxmin())
        .astype("Int32")
    )
    df_oecd["min_ind"] = pd.NA
    df_oecd.loc[:, "min_ind"] = df_oecd.loc[
        df_oecd["argmin"].fillna(0), "industry"
    ].values
    # compute a total
    df_oecd["GDP"] = df_oecd.groupby(["datetime", "LOCATION"])["value"].transform(
        lambda x: x.sum()
    )
    # then shares as a pct
    df_oecd["Percent"] = round(100 * df_oecd["value"] / df_oecd["GDP"], 2)
    return df_oecd


@st.cache
def fred_data_uk_wages():
    """Long run UK average wages from FRED

    :return: wages in dataframe
    :rtype: pandas dataframe
    """
    start = datetime.datetime(1919, 1, 1)
    end = datetime.datetime(2016, 1, 1)
    return web.DataReader("AWEPPUKQ", "fred", start, end)


# ----------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------
st.title("An example dashboard")
# Here's some text...
intro_text = "This is a short example dashboard demonstrating some of the basic functionality of streamlit."
# ...but it only appears in the script when we call `st.write`
st.write(intro_text)
# Here's some markdown
st.markdown("### This is a markdown subtitle")
st.markdown(
    "Regular markdown syntax, inlcuding [links](https://aeturrell.github.io/coding-for-economists), will work. You can use `st.latex` to embed latex equations. Here's the result of using that command:"
)
# Here's an example of a latex equation:
st.latex(r"Y = \beta_0 + \beta_1X + \varepsilon")

st.markdown("### Data: in two columns using `st.columns`")

c1, c2 = st.columns((1, 1))
c1.write("#### OECD Data")
text = "We're going to download data from the OECD API for the UK by *industry*. We can wrap up the data retrieval and cleaning in a function and cache it using `st.cache`; very helpful if you're using an API."
c1.write(text)


df = get_oecd_data()
c1.write("Here are the first few lines of the OECD data:")
c1.write(df.head())
c2.write("#### FRED Data")
c2.write(
    "We're also going to download data from FRED. This will cover historical wages and, later, we'll give the option to plot it on both a log and a linear scale."
)
# Get FRED data on UK wages
fred_uk_awe = fred_data_uk_wages()
c2.write("FRED UK wage data first few lines:")
c2.write(fred_uk_awe.head())

st.markdown("### Metrics")
st.write(
    "Next, we'll use the columns functionality to show which sectors are the biggest movers on the quarter making use of OECD data and industry categories."
)

# want to compute these using the last timestep available in data
# subset to most recent data
subdf = df.loc[df["datetime"] == df["datetime"].max(), :]
rise_ind, rise_growth = subdf["max_ind"].iloc[0], subdf["max"].iloc[0] * 100
rise_share = subdf.loc[subdf["industry"] == rise_ind, "Percent"]
fall_ind, fall_growth = subdf["min_ind"].iloc[0], subdf["min"].iloc[0] * 100
fall_share = subdf.loc[subdf["industry"] == fall_ind, "Percent"]

st.write(
    f"#### For the quarter ending {df['datetime'].max():%Y-%m-%d}, the biggest moves were:"
)

col1, col2 = st.columns(2)
col1.metric("Biggest rise:", f"{rise_ind}", f"{rise_growth:.2f} %")
col2.metric("Biggest fall:", f"{fall_ind}", f"{fall_growth:.2f} %")


st.markdown("### Charts and Interactivity")

st.write(
    "This section will show how to make charts using **altair** and **matplotlib**, two different plotting libraries. In both cases, we have added interactive elements so that the user can change the charts and have them update instantly."
)

st.markdown("#### Altair")
st.write(
    "We're going to plot data on UK output by sector sourced from the OECD. Note that altair charts have some interactivity built in (try zooming in or out, or hovering your mouse)."
)
# Let's give users a multi-select box to choose industries
chosen_industries = st.multiselect(
    "Choose which industries to display", industry_names, default=industry_names
)
# Filter OECD data just to chosen industries
subdf = df[df["industry"].isin(chosen_industries)]

graph = (
    alt.Chart(subdf)
    .mark_bar(size=20)
    .encode(
        alt.Y("value:Q", scale=alt.Scale(domain=(0, 600e3))),
        x="datetime:T",
        color="industry",
        tooltip=["industry", subdf.columns[-1]],
    )
    .properties(
        title="UK output by sector (chained volume, seasonally adjusted; mn GBP)"
    )
    .interactive()
)
st.altair_chart(graph, use_container_width=True)
st.write("Try changing the multi-select box above to see how the chart changes.")


st.markdown("#### Matplotlib")
# Interactive to choose log or linear scale
st.write(
    "This next chart is plotted by **matplotlib**, and doesn't come with any built-in interactivity. However, we have added interactivity using streamlit; the checkbox allows you to show the data on a log or linear scale."
)
logscale = st.checkbox("Log scale?", False)

# Plot chart
fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
ax.plot(fred_uk_awe)
ax.set_ylim(1, None)
ax.set_ylabel("Seasonally adjusted GBP (nominal)")
scale_txt = "linear scale"
if logscale:
    scale_txt = "log scale"
    ax.set_yscale("log")
ax.set_title(f"Average weekly earnings per person in the UK ({scale_txt})", loc="left")
st.pyplot(fig)
