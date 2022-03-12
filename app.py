import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandasdmx as pdmx


# Plot settings
plt.style.use(
    "https://github.com/aeturrell/coding-for-economists/raw/main/plot_style.txt"
)

st.title("An example dashboard")
# Here's some text...
intro_text = "This is a short example dashboard demonstrating the key capabilities of streamlit."
# ...but it only appears in the script when we call `st.write`
st.write(intro_text)

# Here's some markdown
st.markdown("### This is a markdown subtitle")
st.markdown("Regular markdown synthax, such as [links](https://aeturrell.github.io/coding-for-economists), works. You can use `st.latex` to embed latex equations. Here's the result of using that command:")
# Here's an example of a latex equation:
st.latex(r"{\displaystyle {\frac {\partial L}{\partial q^{i}}}(t,{\boldsymbol {q}}(t),{\dot {\boldsymbol {q}}}(t))-{\frac {\mathrm {d} }{\mathrm {d} t}}{\frac {\partial L}{\partial {\dot {q}}^{i}}}(t,{\boldsymbol {q}}(t),{\dot {\boldsymbol {q}}}(t))=0,\quad i=1,\dots ,n.}")


st.markdown("### Data")
text = "We're going to download data from the OECD API for the UK by *industry*. We can wrap up the data retrieval and cleaning in a function and cache it using `st.cache`; very helpful if you're using an API."
st.write(text)

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


@st.cache
def get_oecd_data():
    # Tell pdmx we want OECD data
    oecd = pdmx.Request("OECD")
    # Set out everything about the request in the format specified by the OECD API
    data = oecd.data(
        resource_id="QNA",
        key="GBR."
        + "+".join(codes)
        + ".LNBQRSA.Q/all?startTime=2015",
    ).to_pandas()

    df_oecd = pd.DataFrame(data).reset_index()
    # some clean up operations
    df_oecd["datetime"] = (
        pd.to_datetime(
            df_oecd["TIME_PERIOD"].apply(lambda x: str(x[:4]) + "-" + str(int(x[-1]) * 3)),
            format="%Y-%m",
        )
        + pd.offsets.MonthEnd()
    )
    df_oecd["value"] = df_oecd["value"]
    df_oecd["industry"] = df_oecd["SUBJECT"].map(industry_dict)
    # compute a total
    df_oecd["GDP"] = df_oecd.groupby(["datetime", "LOCATION"])["value"].transform(
        lambda x: x.sum()
    )
    # then shares as a pct
    df_oecd["fraction"] = df_oecd["value"] / df_oecd["GDP"]
    return df_oecd

df = get_oecd_data()

st.write("Data downloaded successfully! First few lines:")
st.write(df.head())

st.markdown("### Metrics")
st.write("Next, we'll use the columns functionality to show which sectors are the biggest movers on the quarter")

# want to compute these using the last timestep available in data
# TODO

col1, col2, col3 = st.columns(3)
col1.metric(f"Biggest rise:", "70 °F", "1.2 °F")
col2.metric(f"Biggest fall:", "9 mph", "-8%")
col3.metric(f"Biggest share:", "86%", "4pp")

st.markdown("### Charts")
st.markdown("#### Altair")
st.write("We're going to try plotting data on UK output by sector sourced from the OECD.")
# Let's give users a multi-select box to choose industries
chosen_industries = st.multiselect("Choose which industries to display",
                                    industry_names,
                                    default=industry_names)
# Filter OECD data just to chosen industries
subdf = df[df['industry'].isin(chosen_industries)]

graph = alt.Chart(subdf).mark_bar(size=15).encode(
    alt.Y('value:Q', scale=alt.Scale(domain=(0, 600e3))),
        x='datetime:T',
        color='industry',
        tooltip=['industry', subdf.columns[-1]]
    ).properties(title="UK output by sector (chained volume, seasonally adjusted; mn GBP)").interactive()
st.altair_chart(graph, use_container_width=True)
st.write("Try changing the multi-select box above to see how the chart changes.")