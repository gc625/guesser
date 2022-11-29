#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")
us_cities = us_cities.query("State in ['New York', 'Ohio']")

import plotly.express as px



# fig = px.line_mapbox(us_cities[:4], lat="lat", lon="lon", color="State", zoom=3, height=300)

# fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=4, mapbox_center_lat = 41,
#     margin={"r":0,"t":0,"l":0,"b":0})

# fig.show()


# %%

best_pred = np.load('best_pred.npy')
best_labels= np.load('best_labels.npy')
worst_pred = np.load('worst_pred.npy')
worst_labels = np.load('worst_labels.npy')


# %%







best_labels

# %%
best_pred
# %%

def plot_results(predictions,labels,type_name='best'):

    color = 'green' if type_name == 'best' else 'red'


    idx = np.array([range(0,len(predictions))]).T
    best_idx = np.array([[f'{type_name}_{i.item()}' for i in idx]]).T

    # best_pred_with_label = np.hstack((best_pred,best_idx))
    # best_labels_with_label = np.hstack((best_labels,best_idx))

    
    # type = np.array([['best' for i in range(len(idx))]]).T
    combined_idx = np.array([[f'{type_name}{i.item()}' for i in idx]]).T
    pred_name = np.array([[f'prediction{i}' for i in range(len(idx))]]).T
    label_name = np.array([[f'label{i}' for i in range(len(idx))]]).T



    predictions_info = np.hstack((predictions,pred_name,idx,combined_idx))
    labels_info = np.hstack((labels,label_name,idx,combined_idx))


    # DATAFRAME FOR PLOTTING LOCATIONS 
    locations_df = pd.DataFrame(predictions_info, columns=['lat','lon','type','idx','marker'])
    locations_df = locations_df.append(pd.DataFrame(labels_info, columns=['lat','lon','type','idx','marker']))
    locations_df = locations_df.astype({'lat': 'float','lon':'float','idx':'int'})



    line_df = pd.DataFrame(np.hstack((predictions,labels)),columns=['start_lat','start_lon','end_lat','end_lon'])


    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = locations_df['lon'],
        lat = locations_df['lat'],
        hoverinfo = 'text',
        text = locations_df['type'],
        mode = 'markers',
        marker = dict(
            size = 10,
            color = color,
            line = dict(
                width = 3,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))

    fig.update_layout(
        title_text = f'{type_name} {len(predictions)} predictions',
        showlegend = False,
        geo = dict(
            scope = 'north america',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    )


    for i in range(len(line_df)):
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'USA-states',
                lon = [line_df['start_lon'][i], line_df['end_lon'][i]],
                lat = [line_df['start_lat'][i], line_df['end_lat'][i]],
                mode = 'lines',
                line = dict(width = 1,color = color),
                # opacity = float(df_flight_paths['cnt'][i]) / float(df_flight_paths['cnt'].max()),
            )
        )

    fig.show()
# %%
plot_results(worst_pred,worst_labels,type_name='worst')

# %%
