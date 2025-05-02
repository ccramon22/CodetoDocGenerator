import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import torch

from main import main as run_training
from model import load_model_and_tokenizer
from evaluation import generate_documentation

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "ParameterPilots"

model_cache = {"model": None, "tokenizer": None}

app.layout = dbc.Container([
    html.H2("ParameterPilots"),
    dcc.Tabs(id="tabs", value="train", children=[
        dcc.Tab(label="Train Model", value="train"),
        dcc.Tab(label="Generate Doc", value="generate")
    ]),
    html.Div(id="tab-content")
], fluid=True)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "train":
        return html.Div([
            dbc.Button("Start Training", id="train-btn", color="primary"),
            html.Div(id="train-output", className="mt-3")
        ])
    return html.Div([
        dcc.Textarea(id="code-input", style={"width": "100%", "height": 200}),
        dbc.Button("Generate", id="generate-btn", color="success", className="mt-2"),
        html.Div(id="doc-output", className="mt-3", style={"whiteSpace": "pre-wrap"})
    ])


@app.callback(Output("train-output", "children"), Input("train-btn", "n_clicks"), prevent_initial_call=True)
def train_model(n):
    try:
        run_training()
        return "Training complete"
    except Exception as e:
        return f"Training failed:\n{e}"


@app.callback(Output("doc-output", "children"), Input("generate-btn", "n_clicks"), State("code-input", "value"), prevent_initial_call=True)
def generate_doc(n, code):
    if not code:
        return "paste a Python function"

    try:
        if not model_cache["model"]:
            model_cache["model"], model_cache["tokenizer"] = load_model_and_tokenizer("./mistral_documentation_generator")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return generate_documentation(code, model_cache["model"].to(device), model_cache["tokenizer"], device=device)
    except Exception as e:
        return f"Generation failed:\n{e}"


if __name__ == "__main__":
    app.run(debug=True)
