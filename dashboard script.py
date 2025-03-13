! pip install dash

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from wordcloud import WordCloud
import base64
from io import BytesIO
from collections import Counter
import plotly.graph_objects as go
import numpy as np

# load processed data, not original csv file
df = pd.read_pickle("processed_texts.pkl")
df["Category"] = df["label"].map({1: "AI", 0: "Human"})

# ensure 'topic' column exists: pervasive issue
if "topic" not in df.columns:
    print("Reapplying topic assignment...")

    topic_keywords = {
        "Politics": ["government", "policy", "election", "electoral", "senator", "vote", "president", "congress"],
        "Space Exploration": ["nasa", "moon", "mars", "spaceship", "astronaut", "venus", "spacecraft"],
        "Education": ["student", "school", "subjects", "college", "university", "learning", "teacher", "classroom", "extracurricular"],
        "Automotive Vehicles": ["car", "driverless", "engine", "road", "fuel", "driving", "highway", "transport"],
        "Motivation": ["success", "goal", "achievement", "work hard", "failure", "positive", "dream", "inspire"],
        "Technology": ["smartphone", "online", "cloud", "software", "computer", "digital", "communication", "cell phone", "internet"]
    }

    def assign_topic(text):
        for topic, keywords in topic_keywords.items():
            if any(word in text for word in keywords):
                return topic
        return None

    df["topic"] = df["processed_text"].apply(assign_topic)

    df["topic"] = df["topic"].fillna(df["label"].map({
        0: "Politics",
        1: "Space Exploration",
        2: "Education",
        3: "Automotive Vehicles",
        4: "Motivation",
        5: "Technology"
    }))

# Topic distribution
topic_counts = df.groupby(["topic", "Category"]).size().reset_index(name="Count")

# Calculate percentages
total_count = topic_counts["Count"].sum()
topic_counts["Percentage"] = (topic_counts["Count"] / total_count) * 100
topic_counts["PercentageLabel"] = topic_counts["Percentage"].round(1).astype(str) + "%"

# Compute the weighted midpoint for balanced coloring
color_midpoint = np.average(topic_counts["Count"], weights=topic_counts["Count"])

# Generate topic-based word clouds
def generate_topic_wordcloud(topic):
    topic_text = " ".join(df[df["topic"] == topic]["processed_text"])
    wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(topic_text)  # Increased size
    buf = BytesIO()
    wordcloud.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Word frequency difference visualization
ai_texts = " ".join(df[df["Category"] == "AI"]["processed_text"])
human_texts = " ".join(df[df["Category"] == "Human"]["processed_text"])

ai_word_freq = Counter(ai_texts.split())
human_word_freq = Counter(human_texts.split())

# Generate word frequency difference data
## Define key words for word frequency difference visualization
key_words = [
    "opinion", "feel", "experience", "understand", "believe",
    "analysis", "findings", "research", "therefore", "hypothesis",
    "perspective", "evidence", "interpret", "hope",
    "conclusion", "fact", "objective", "significant"
]

# Generate word frequency difference data
word_freq_df = pd.DataFrame({
    "word": key_words,
    "ai_freq": [ai_word_freq.get(word, 0) for word in key_words],
    "human_freq": [human_word_freq.get(word, 0) for word in key_words]
})

# Ensure word column formatting
word_freq_df["word"] = word_freq_df["word"].str.title()
word_freq_df["freq_diff"] = word_freq_df["human_freq"] - word_freq_df["ai_freq"]
word_freq_df = word_freq_df[(word_freq_df["ai_freq"] > 0) | (word_freq_df["human_freq"] > 0)]
word_freq_df = word_freq_df.sort_values(by="freq_diff", ascending=True)



# Dash App
app = dash.Dash(__name__)

app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "padding": "20px", "backgroundColor": "#f8f9fa"}, children=[
    html.H1("AI vs. Human Writing Analysis", style={"textAlign": "center", "color": "#333", "fontSize": "32px"}),

    # First Bar Chart
    html.Div([
        html.H3("What Do Humans Write About vs. What They Use AI For?", style={"textAlign": "center", "color": "#444"}),
        html.P("This visualization was based on the categorization of AI and Human essays and the analysis of their topic distribution to see if there are any differences in trends.",
               style={"textAlign": "center", "color": "#555", "fontSize": "14px", "maxWidth": "800px", "margin": "auto"}),
        dcc.Graph(
            figure=px.bar(
                topic_counts,
                x="topic", y="Count", color="Category", barmode="group",
                color_discrete_map={"Human": "#70A494", "AI": "#E8998D"}
            ).update_layout(paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa", font=dict(color="#333"))
        ),
    ], style={"marginBottom": "40px"}),

    # Treemap
    html.Div([
        html.H3("What Do Humans Write About vs. What They Use AI For?", style={"textAlign": "center", "color": "#444"}),
        html.P("This is a treemap visualization of the same data above.",
               style={"textAlign": "center", "color": "#555", "fontSize": "14px", "maxWidth": "800px", "margin": "auto"}),
        dcc.Graph(
            figure=px.treemap(
                topic_counts,
                path=[px.Constant("All Topics"), "Category", "topic"],
                values="Count",
                color="Count",
                hover_data={"Count": True, "Percentage": True},
                color_continuous_scale="RdBu",
                color_continuous_midpoint=color_midpoint
            ).update_traces(textinfo="label+text", text=topic_counts["PercentageLabel"])
            .update_layout(
                title_x=0.5,
                paper_bgcolor="#f8f9fa",
                font=dict(color="#333")
            )
        )
    ], style={"marginBottom": "40px"}),

    # Word Clouds by Topic
    html.H3("Word Clouds by Topic", style={"textAlign": "center", "color": "#444"}),
    html.Div([
        html.Div([
            html.H4(topic, style={"textAlign": "center", "color": "#555"}),
            html.Img(src="data:image/png;base64,{}".format(generate_topic_wordcloud(topic)), style={"width": "45%", "display": "block", "margin": "auto"})
        ], style={"display": "inline-block", "padding": "20px"}) for topic in df["topic"].unique()
    ], style={"textAlign": "center"}),

    # Word Frequency Difference Bar Chart
    html.Div([
        html.H3("Word Usage Difference Between AI and Human Essays", style={"textAlign": "center", "color": "#444"}),
        html.P("For this visualization, I chose words that I thought were meaningful in the context of AI-written and human-written work in my experience.",
               style={"textAlign": "center", "color": "#555", "fontSize": "14px", "maxWidth": "800px", "margin": "auto"}),
        dcc.Graph(
            figure=go.Figure([
                go.Bar(x=word_freq_df[word_freq_df["freq_diff"] > 0]["freq_diff"], y=word_freq_df[word_freq_df["freq_diff"] > 0]["word"],
                       orientation="h", marker=dict(color="red"), name="More Common in Human Essays"),
                go.Bar(x=word_freq_df[word_freq_df["freq_diff"] < 0]["freq_diff"], y=word_freq_df[word_freq_df["freq_diff"] < 0]["word"],
                       orientation="h", marker=dict(color="blue"), name="More Common in AI Essays")
            ]).update_layout(
                title_x=0.5,
                paper_bgcolor="white",
                plot_bgcolor="#E5ECF6",
                font=dict(color="#333")
            )
        )
    ], style={"marginTop": "40px"})
])

if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1", port=8050)