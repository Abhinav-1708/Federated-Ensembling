import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Row, Col, Card, Button, Form, ProgressBar, Badge, Alert } from 'react-bootstrap';
import axios from 'axios';

const ModelBadge = ({ modelType }) => {
  const models = ['Random Forest', 'Gradient Boosting', 'Neural Network'];
  const classes = [`model-badge-${modelType}`, 'text-white'];
  
  return (
    <Badge className={classes.join(' ')} pill>
      {models[modelType]}
    </Badge>
  );
};

const ClientCard = ({ clientId, status, currentModel }) => {
  const statusClass = `client-card client-${status}`;
  
  return (
    <Card className={`mb-2 ${statusClass}`}>
      <Card.Body className="p-3">
        <Row>
          <Col xs={4}>
            <h5 className="mb-0">Client {clientId}</h5>
          </Col>
          <Col xs={4} className="text-center">
            <Badge bg={
              status === 'waiting' ? 'secondary' :
              status === 'training' ? 'primary' :
              status === 'completed' ? 'success' : 'danger'
            }>
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </Badge>
          </Col>
          <Col xs={4} className="text-end">
            {currentModel !== null && status !== 'waiting' && (
              <ModelBadge modelType={currentModel} />
            )}
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

const TrainingControl = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    rounds: 10,
    clients: 3,
    models: 3,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState({
    current_round: 0,
    total_rounds: 0,
    client_status: {},
    latest_run_id: null,
  });
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  
  const statusInterval = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    // Check if training is already in progress when the component loads
    const checkTrainingStatus = async () => {
      try {
        const response = await axios.get('/api/training/status');
        if (response.data.status.is_running) {
          setIsTraining(true);
          setTrainingStatus(response.data.status);
          setLogs(response.data.logs);
          
          // Start polling for updates
          startStatusPolling();
        }
      } catch (err) {
        console.error('Error checking training status:', err);
      }
    };
    
    checkTrainingStatus();
    
    // Clear interval on component unmount
    return () => {
      if (statusInterval.current) {
        clearInterval(statusInterval.current);
      }
    };
  }, []);
  
  // Scroll to bottom of logs when they update
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);
  
  const startStatusPolling = () => {
    if (statusInterval.current) {
      clearInterval(statusInterval.current);
    }
    
    statusInterval.current = setInterval(async () => {
      try {
        const response = await axios.get('/api/training/status');
        setTrainingStatus(response.data.status);
        setLogs(response.data.logs);
        
        // If training has completed
        if (!response.data.status.is_running && isTraining) {
          setIsTraining(false);
          clearInterval(statusInterval.current);
          
          // If we have a run ID, we can navigate to the results
          if (response.data.status.latest_run_id) {
            setTimeout(() => {
              navigate(`/run/${response.data.status.latest_run_id}`);
            }, 3000);
          }
        }
      } catch (err) {
        console.error('Error polling training status:', err);
      }
    }, 1000);
  };
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: parseInt(value, 10) });
  };
  
  const handleStartTraining = async () => {
    try {
      setError(null);
      const response = await axios.post('/api/training/start', formData);
      setIsTraining(true);
      
      // Start polling for status updates
      startStatusPolling();
    } catch (err) {
      console.error('Error starting training:', err);
      setError(err.response?.data?.error || 'Failed to start training. Please try again.');
    }
  };
  
  const handleStopTraining = async () => {
    try {
      await axios.post('/api/training/stop');
      setIsTraining(false);
      
      if (statusInterval.current) {
        clearInterval(statusInterval.current);
        statusInterval.current = null;
      }
    } catch (err) {
      console.error('Error stopping training:', err);
    }
  };
  
  return (
    <div>
      <Row className="mb-4">
        <Col>
          <h2>Training Control</h2>
          <p className="lead">
            Configure and start a new federated ensemble learning training session.
          </p>
        </Col>
      </Row>
      
      {error && (
        <Alert variant="danger" onClose={() => setError(null)} dismissible>
          {error}
        </Alert>
      )}
      
      <Row>
        <Col md={5}>
          <Card className="mb-4">
            <Card.Header>
              <h4>Training Configuration</h4>
            </Card.Header>
            <Card.Body>
              <Form>
                <Form.Group className="mb-3">
                  <Form.Label>Number of Rounds</Form.Label>
                  <Form.Control
                    type="number"
                    name="rounds"
                    value={formData.rounds}
                    onChange={handleInputChange}
                    disabled={isTraining}
                    min="1"
                    max="50"
                  />
                  <Form.Text className="text-muted">
                    Number of federated training rounds (1-50)
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Number of Clients</Form.Label>
                  <Form.Control
                    type="number"
                    name="clients"
                    value={formData.clients}
                    onChange={handleInputChange}
                    disabled={isTraining}
                    min="2"
                    max="10"
                  />
                  <Form.Text className="text-muted">
                    Number of federated clients (2-10)
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Number of Models</Form.Label>
                  <Form.Control
                    type="number"
                    name="models"
                    value={formData.models}
                    onChange={handleInputChange}
                    disabled={isTraining}
                    min="1"
                    max="3"
                  />
                  <Form.Text className="text-muted">
                    Number of models in the ensemble (1-3)
                  </Form.Text>
                </Form.Group>
                
                {!isTraining ? (
                  <Button 
                    variant="success" 
                    size="lg" 
                    className="w-100"
                    onClick={handleStartTraining}
                  >
                    Start Training
                  </Button>
                ) : (
                  <Button 
                    variant="danger" 
                    size="lg" 
                    className="w-100"
                    onClick={handleStopTraining}
                  >
                    Stop Training
                  </Button>
                )}
              </Form>
            </Card.Body>
          </Card>
          
          {isTraining && (
            <Card>
              <Card.Header>
                <h4>Training Progress</h4>
              </Card.Header>
              <Card.Body>
                <p className="mb-2">
                  <strong>Round:</strong> {trainingStatus.current_round} / {trainingStatus.total_rounds}
                </p>
                
                <ProgressBar 
                  now={(trainingStatus.current_round / trainingStatus.total_rounds) * 100} 
                  label={`${trainingStatus.current_round}/${trainingStatus.total_rounds}`}
                  className="mb-4 round-progress"
                />
                
                <h5>Client Status:</h5>
                {Object.entries(trainingStatus.client_status).map(([clientId, status]) => (
                  <ClientCard 
                    key={clientId}
                    clientId={clientId}
                    status={status.status}
                    currentModel={status.current_model}
                  />
                ))}
                
                {trainingStatus.latest_run_id && (
                  <Alert variant="success" className="mt-3">
                    Training completed! Redirecting to results...
                  </Alert>
                )}
              </Card.Body>
            </Card>
          )}
        </Col>
        
        <Col md={7}>
          <Card>
            <Card.Header>
              <h4>Training Logs</h4>
            </Card.Header>
            <Card.Body className="p-0">
              <div className="logs-container">
                {logs.length === 0 ? (
                  <p className="log-line p-3">No logs available. Start training to see logs here.</p>
                ) : (
                  logs.map((log, index) => (
                    <p key={index} className="log-line">
                      {log}
                    </p>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TrainingControl; 