import pandas as pd
import plotly.graph_objs as go

# Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
# `data/file_name.csv`

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots arable land from 1990 to 2015 in top 10 economies
    # as a line chart
    df = pd.read_csv("data/starbucks.csv")

    graph_one = []
    x_val = df[df["purchase"] == 1]["V2"].tolist()
    y_val = df[df["purchase"] == 1]["V3"].tolist()
    graph_one.append(
      go.Scatter(
      x = x_val,
      y = y_val,
      mode = 'markers'
      )
    )

    layout_one = dict(title = 'Column "V2" vs "V3" in Purchase Group',
                xaxis = dict(title = 'V2'),
                yaxis = dict(title = 'V3'),
                )

# second chart plots ararble land for 2015 as a bar chart
    graph_two = []
    x_val = df[df["purchase"] == 0]["V2"].tolist()
    y_val = df[df["purchase"] == 0]["V3"].tolist()
    graph_two.append(
      go.Scatter(
      x = x_val,
      y = y_val,
      mode = 'markers'
      )
    )

    layout_two = dict(title = 'Column "V2" vs "V3" in Non-Purchase Group',
                xaxis = dict(title = 'V2'),
                yaxis = dict(title = 'V3'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []

    labels = df[df["purchase"] == 1].V5.value_counts().index.tolist()
    values = df[df["purchase"] == 1].V5.value_counts().values.tolist()
    colors = ['#BFB3FE', '#ee544c', '#1a81f4', '#D0F9B1']
    graph_three.append(
      go.Pie(labels=labels, values=values,
                     hoverinfo='label+percent+name', textinfo='percent',
                     textfont=dict(size=20),
                     marker=dict(colors = colors,
                     line=dict(color='#000000', width=2)))
    )

    layout_three = dict(title = 'Column "V5" Distribution in Purchase Group')


# fourth chart shows rural population vs arable land
    graph_four = []

    labels = df[df["purchase"] == 0].V5.value_counts().index.tolist()
    values = df[df["purchase"] == 0].V5.value_counts().values.tolist()
    colors = ['#BFB3FE', '#ee544c', '#1a81f4', '#D0F9B1']
    graph_four.append(
      go.Pie(labels=labels, values=values,
                     hoverinfo='label+percent+name', textinfo='percent',
                     textfont=dict(size=20),
                     marker=dict(colors = colors,
                     line=dict(color='#000000', width=2)))
    )

    layout_four = dict(title = 'Column "V5" Distribution in Purchase Group')

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures
