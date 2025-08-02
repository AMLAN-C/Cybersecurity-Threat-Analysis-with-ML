import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Load and preprocess dataset
df = pd.read_csv("data/CloudWatch_Traffic_Web_Attack.csv")
df['creation_time'] = pd.to_datetime(df['creation_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['time'] = pd.to_datetime(df['time'])
df['session_duration'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
df['byte_ratio'] = df['bytes_out'] / (df['bytes_in'] + 1)

# Run Isolation Forest
features = ['bytes_in', 'bytes_out', 'session_duration', 'byte_ratio']
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = iso.fit_predict(df[features])
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# Filter anomalies
anomalies = df[df['is_anomaly'] == 1]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Web Threat Anomaly Dashboard", style={'textAlign': 'center'}),

    dcc.Graph(
        figure=px.scatter(
            df, x='bytes_in', y='bytes_out',
            color=df['is_anomaly'].map({1: 'Anomaly', 0: 'Normal'}),
            title="Bytes In vs Bytes Out (Anomaly Detection)",
            labels={'color': 'Traffic Type'},
            hover_data=['src_ip', 'src_ip_country_code', 'dst_port']
        )
    ),

    dcc.Graph(
        figure=px.histogram(
            anomalies, x='src_ip_country_code',
            title="Top Source Countries (Anomalous Traffic)",
            color_discrete_sequence=['indianred']
        )
    ),

    dcc.Graph(
        figure=px.scatter(
            anomalies, x='session_duration', y='byte_ratio',
            title="Session Duration vs Byte Ratio (Anomalies Only)",
            hover_data=['src_ip', 'dst_port'],
            color='src_ip_country_code'
        )
    ),

    dcc.Graph(
        figure=px.bar(
            anomalies['dst_port'].value_counts().reset_index(),
            x='index', y='dst_port',
            title="Destination Ports Used in Anomalies",
            labels={'index': 'Port', 'dst_port': 'Count'}
        )
    )
])

if __name__ == "__main__":
    app.run(debug=True)
