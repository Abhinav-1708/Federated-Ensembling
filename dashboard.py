import os
import json
import glob
import subprocess
import threading
import time
from flask import Flask, render_template, send_from_directory, jsonify, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='./frontend/build', static_url_path='/')
CORS(app)  # Enable CORS for all routes

# Global variables to track training status
training_process = None
training_logs = []
training_status = {
    "is_running": False,
    "current_round": 0,
    "total_rounds": 0,
    "client_status": {},
    "latest_run_id": None
}

@app.route('/')
def serve():
    """Serve React app"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/results/<run_id>')
def results(run_id):
    """Display results for a specific run"""
    try:
        # Load results file
        results_path = f'results/results_{run_id}.json'
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Load ensemble weights
        weights_path = f'saved_models/ensemble_weights_{run_id}.npy'
        if os.path.exists(weights_path):
            weights = np.load(weights_path).tolist()
        else:
            weights = None
        
        # Get model metrics from the last round
        last_round = results_data[-1] if results_data else {}
        
        # Get plots
        accuracy_plot = f'/plot/accuracy/{run_id}'
        weights_plot = f'/plot/weights/{run_id}'
        
        return render_template(
            'results.html', 
            run_id=run_id,
            results=results_data,
            last_round=last_round,
            weights=weights,
            accuracy_plot=accuracy_plot,
            weights_plot=weights_plot
        )
    except Exception as e:
        return f"Error loading results: {str(e)}"

@app.route('/plot/<plot_type>/<run_id>')
def get_plot(plot_type, run_id):
    """Return plots as images"""
    if plot_type == 'accuracy':
        plot_path = f'results/accuracy_plot_{run_id}.png'
    elif plot_type == 'weights':
        plot_path = f'results/weights_plot_{run_id}.png'
    else:
        return "Invalid plot type"
    
    if os.path.exists(plot_path):
        directory = os.path.dirname(plot_path)
        filename = os.path.basename(plot_path)
        return send_from_directory(directory, filename)
    else:
        # Generate plot dynamically if file doesn't exist
        if plot_type == 'accuracy':
            return generate_accuracy_plot(run_id)
        elif plot_type == 'weights':
            return generate_weights_plot(run_id)

def generate_accuracy_plot(run_id):
    """Generate accuracy plot dynamically"""
    try:
        results_path = f'results/results_{run_id}.json'
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Extract accuracy data for each model and ensemble
        rounds = range(1, len(results_data) + 1)
        model_0_acc = [round_data.get('model_0', {}).get('accuracy', 0) for round_data in results_data]
        model_1_acc = [round_data.get('model_1', {}).get('accuracy', 0) for round_data in results_data]
        model_2_acc = [round_data.get('model_2', {}).get('accuracy', 0) for round_data in results_data]
        ensemble_acc = [round_data.get('ensemble', {}).get('accuracy', 0) for round_data in results_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, model_0_acc, 'o-', label='Random Forest')
        plt.plot(rounds, model_1_acc, 's-', label='Gradient Boosting')
        plt.plot(rounds, model_2_acc, '^-', label='Neural Network')
        plt.plot(rounds, ensemble_acc, 'D-', label='Ensemble', linewidth=2)
        
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy over Training Rounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        buf.seek(0)
        return send_from_directory('results', f'accuracy_plot_{run_id}.png')
    except Exception as e:
        return f"Error generating accuracy plot: {str(e)}"

def generate_weights_plot(run_id):
    """Generate weights plot dynamically"""
    try:
        results_path = f'results/results_{run_id}.json'
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Extract weights data
        rounds = range(1, len(results_data) + 1)
        weights_0 = [round_data.get('ensemble', {}).get('weights', [0.33, 0.33, 0.33])[0] for round_data in results_data]
        weights_1 = [round_data.get('ensemble', {}).get('weights', [0.33, 0.33, 0.33])[1] for round_data in results_data]
        weights_2 = [round_data.get('ensemble', {}).get('weights', [0.33, 0.33, 0.33])[2] for round_data in results_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, weights_0, 'o-', label='Random Forest')
        plt.plot(rounds, weights_1, 's-', label='Gradient Boosting')
        plt.plot(rounds, weights_2, '^-', label='Neural Network')
        
        plt.xlabel('Round')
        plt.ylabel('Weight')
        plt.title('Ensemble Weights over Training Rounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        buf.seek(0)
        return send_from_directory('results', f'weights_plot_{run_id}.png')
    except Exception as e:
        return f"Error generating weights plot: {str(e)}"

@app.route('/api/runs')
def api_runs():
    """API endpoint to get list of runs"""
    result_files = sorted(glob.glob('results/results_*.json'), reverse=True)
    runs = []
    
    for result_file in result_files:
        run_id = os.path.basename(result_file).replace('results_', '').replace('.json', '')
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                last_round = data[-1] if data else {}
                
                # Get final metrics
                ensemble_acc = last_round.get('ensemble', {}).get('accuracy', 0)
                
                runs.append({
                    'id': run_id,
                    'timestamp': int(run_id) if run_id.isdigit() else 0,
                    'rounds': len(data),
                    'ensemble_accuracy': ensemble_acc
                })
        except Exception as e:
            print(f"Error loading results file {result_file}: {str(e)}")
            continue
    
    return jsonify(runs)

@app.route('/api/run/<run_id>')
def api_run_details(run_id):
    """API endpoint to get details for a specific run"""
    try:
        # Load results file
        results_path = f'results/results_{run_id}.json'
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        # Load ensemble weights if they exist
        weights_path = f'saved_models/ensemble_weights_{run_id}.npy'
        if os.path.exists(weights_path):
            weights = np.load(weights_path).tolist()
        else:
            weights = None
        
        return jsonify({
            'run_id': run_id,
            'results': results_data,
            'last_round': results_data[-1] if results_data else {},
            'weights': weights
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/plot/<plot_type>/<run_id>')
def api_get_plot(plot_type, run_id):
    """API endpoint to get plot URLs for a specific run"""
    if plot_type == 'accuracy':
        plot_path = f'results/accuracy_plot_{run_id}.png'
    elif plot_type == 'weights':
        plot_path = f'results/weights_plot_{run_id}.png'
    else:
        return jsonify({"error": "Invalid plot type"}), 400
    
    if os.path.exists(plot_path):
        return jsonify({"url": f"/plot-image/{plot_type}/{run_id}"})
    else:
        # Try generating the plot
        try:
            if plot_type == 'accuracy':
                generate_accuracy_plot(run_id)
            elif plot_type == 'weights':
                generate_weights_plot(run_id)
            
            return jsonify({"url": f"/plot-image/{plot_type}/{run_id}"})
        except:
            return jsonify({"error": "Could not generate plot"}), 400

@app.route('/plot-image/<plot_type>/<run_id>')
def get_plot_image(plot_type, run_id):
    """Return plots as images"""
    if plot_type == 'accuracy':
        plot_path = f'results/accuracy_plot_{run_id}.png'
    elif plot_type == 'weights':
        plot_path = f'results/weights_plot_{run_id}.png'
    else:
        return "Invalid plot type", 400
    
    if os.path.exists(plot_path):
        directory = os.path.dirname(plot_path)
        filename = os.path.basename(plot_path)
        return send_from_directory(directory, filename)
    else:
        # Generate plot dynamically if file doesn't exist
        if plot_type == 'accuracy':
            return generate_accuracy_plot(run_id)
        elif plot_type == 'weights':
            return generate_weights_plot(run_id)

@app.route('/api/download/weights/<run_id>')
def download_weights(run_id):
    """API endpoint to download weights file"""
    weights_path = f'saved_models/ensemble_weights_{run_id}.npy'
    if os.path.exists(weights_path):
        directory = os.path.dirname(weights_path)
        filename = os.path.basename(weights_path)
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({"error": "Weights file not found"}), 404

@app.route('/api/download/model/<model_idx>/<run_id>')
def download_model(model_idx, run_id):
    """API endpoint to download model file"""
    if model_idx == '2':  # Neural Network
        model_path = f'saved_models/model_{model_idx}_{run_id}.h5'
    else:  # Sklearn models
        model_path = f'saved_models/model_{model_idx}_{run_id}.pkl'
    
    if os.path.exists(model_path):
        directory = os.path.dirname(model_path)
        filename = os.path.basename(model_path)
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({"error": "Model file not found"}), 404

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """API endpoint to start the training process"""
    global training_process, training_logs, training_status
    
    if training_status["is_running"]:
        return jsonify({"error": "Training is already running"}), 400
    
    # Get parameters from request
    data = request.json
    rounds = data.get('rounds', 10)
    clients = data.get('clients', 3)
    models = data.get('models', 3)
    
    # Reset training state
    training_logs = []
    training_status = {
        "is_running": True,
        "current_round": 0,
        "total_rounds": rounds,
        "client_status": {i: {"status": "waiting", "current_model": None} for i in range(clients)},
        "latest_run_id": None
    }
    
    # Create a thread to run the training process
    thread = threading.Thread(target=run_training_process, args=(rounds, clients, models))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "Training started", "rounds": rounds, "clients": clients, "models": models})

def run_training_process(rounds, clients, models):
    """Run the training process in a separate thread"""
    global training_process, training_logs, training_status
    
    try:
        cmd = f"python main.py --rounds {rounds} --clients {clients} --models {models}"
        
        # Start the process and capture output
        training_process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Monitor output to track progress
        for line in iter(training_process.stdout.readline, ''):
            training_logs.append(line.strip())
            
            # Parse the output to update training status
            if "Round" in line and "/" in line:
                try:
                    # Extract current round from output like "Round 2/10"
                    parts = line.split("Round ")[1].split("/")
                    current_round = int(parts[0].strip())
                    training_status["current_round"] = current_round
                except:
                    pass
            
            # Track client status
            if "Client" in line and "training" in line:
                try:
                    # Extract client_id and model_idx
                    for i in range(clients):
                        if f"Client {i}" in line:
                            model_type = None
                            if "Random Forest" in line:
                                model_type = 0
                            elif "Gradient Boosting" in line:
                                model_type = 1
                            elif "Neural Network" in line:
                                model_type = 2
                            
                            if model_type is not None:
                                training_status["client_status"][i] = {
                                    "status": "training",
                                    "current_model": model_type
                                }
                except:
                    pass
            
            # Track when clients complete training
            if "Client" in line and "completed training" in line:
                try:
                    for i in range(clients):
                        if f"Client {i}" in line:
                            training_status["client_status"][i]["status"] = "completed"
                except:
                    pass
            
            # Identify the run ID when results are saved
            if "Results saved to" in line:
                try:
                    results_path = line.split("Results saved to ")[1].strip()
                    file_name = os.path.basename(results_path)
                    run_id = file_name.replace("results_", "").replace(".json", "")
                    training_status["latest_run_id"] = run_id
                except:
                    pass
        
        training_process.wait()
        
    except Exception as e:
        training_logs.append(f"Error: {str(e)}")
    finally:
        training_status["is_running"] = False
        training_process = None

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """API endpoint to stop the training process"""
    global training_process, training_status
    
    if training_process and training_status["is_running"]:
        training_process.terminate()
        training_status["is_running"] = False
        return jsonify({"status": "Training stopped"})
    else:
        return jsonify({"error": "No training process running"}), 400

@app.route('/api/training/status')
def get_training_status():
    """API endpoint to get the current training status"""
    global training_status, training_logs
    
    return jsonify({
        "status": training_status,
        "logs": training_logs[-50:] if training_logs else []  # Return only the last 50 logs to avoid excessive data
    })

# Create templates directory and templates if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create index.html template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Federated Ensemble Learning Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 2rem; }
        .run-item { cursor: pointer; }
        .run-item:hover { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Federated Ensemble Learning Dashboard</h1>
        
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Available Runs</h5>
                    </div>
                    <div class="card-body">
                        <div class="list-group">
                            {% if run_ids %}
                                {% for run_id in run_ids %}
                                    <a href="/results/{{ run_id }}" class="list-group-item list-group-item-action run-item">
                                        Run {{ run_id }}
                                    </a>
                                {% endfor %}
                            {% else %}
                                <p>No runs available. Run the federated learning system first.</p>
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header">
                        <h5>Actions</h5>
                    </div>
                    <div class="card-body">
                        <a href="/" class="btn btn-primary">Refresh</a>
                        <a href="#" onclick="window.location.href='/';" class="btn btn-secondary">Home</a>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="jumbotron">
                    <h2>Federated Ensemble Learning</h2>
                    <p class="lead">
                        This dashboard displays results from federated ensemble learning runs. Select a run from the list to view detailed results.
                    </p>
                    <hr class="my-4">
                    <p>Run the main.py script to generate new results:</p>
                    <pre class="bg-light p-3">python main.py --rounds 10 --clients 3 --models 3</pre>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''')

# Create results.html template
with open('templates/results.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Run Results - Federated Ensemble Learning</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 2rem; }
        .metric-card { margin-bottom: 1rem; }
        .metric-value { font-size: 1.5rem; font-weight: bold; }
        .plot-container { margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Run Results <small class="text-muted">{{ run_id }}</small></h1>
        
        <div class="row mb-4">
            <div class="col-12">
                <a href="/" class="btn btn-primary">Back to Dashboard</a>
                <a href="/results/{{ run_id }}" class="btn btn-secondary">Refresh</a>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <!-- Plots Section -->
                <div class="card plot-container">
                    <div class="card-header">
                        <h5>Performance Over Time</h5>
                    </div>
                    <div class="card-body">
                        <img src="{{ accuracy_plot }}" class="img-fluid" alt="Accuracy Plot">
                    </div>
                </div>
                
                <div class="card plot-container">
                    <div class="card-header">
                        <h5>Ensemble Weights</h5>
                    </div>
                    <div class="card-body">
                        <img src="{{ weights_plot }}" class="img-fluid" alt="Weights Plot">
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <!-- Final Metrics -->
                <div class="card mb-4">
                    <div class="card-header">
                        <h5>Final Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-sm-6 metric-card">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h6>Random Forest</h6>
                                        <div class="metric-value">
                                            {{ "%.4f"|format(last_round.model_0.accuracy) if last_round.model_0 else "N/A" }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-sm-6 metric-card">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h6>Gradient Boosting</h6>
                                        <div class="metric-value">
                                            {{ "%.4f"|format(last_round.model_1.accuracy) if last_round.model_1 else "N/A" }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-sm-6 metric-card">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h6>Neural Network</h6>
                                        <div class="metric-value">
                                            {{ "%.4f"|format(last_round.model_2.accuracy) if last_round.model_2 else "N/A" }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-sm-6 metric-card">
                                <div class="card bg-primary text-white">
                                    <div class="card-body text-center">
                                        <h6>Ensemble</h6>
                                        <div class="metric-value">
                                            {{ "%.4f"|format(last_round.ensemble.accuracy) if last_round.ensemble else "N/A" }}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Ensemble Weights -->
                <div class="card">
                    <div class="card-header">
                        <h5>Final Ensemble Weights</h5>
                    </div>
                    <div class="card-body">
                        {% if weights %}
                            <ul class="list-group">
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Random Forest
                                    <span class="badge badge-primary badge-pill">{{ "%.4f"|format(weights[0]) }}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Gradient Boosting
                                    <span class="badge badge-primary badge-pill">{{ "%.4f"|format(weights[1]) }}</span>
                                </li>
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    Neural Network
                                    <span class="badge badge-primary badge-pill">{{ "%.4f"|format(weights[2]) }}</span>
                                </li>
                            </ul>
                        {% else %}
                            <p>No weights available</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Create frontend directory structure for React
    os.makedirs('frontend', exist_ok=True)
    os.makedirs('frontend/build', exist_ok=True)
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 