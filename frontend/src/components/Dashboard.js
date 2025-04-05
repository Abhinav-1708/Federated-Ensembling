import React, { useState, useEffect } from 'react';
import { Row, Col, Card, ListGroup, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import axios from 'axios';

const Dashboard = () => {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/api/runs');
        setRuns(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching runs:', err);
        setError('Failed to fetch runs. Please try again later.');
        setLoading(false);
      }
    };

    fetchRuns();
  }, []);

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  return (
    <div>
      <Row className="mb-4">
        <Col>
          <h2>Federated Ensemble Learning Dashboard</h2>
          <p className="lead">
            View results from previous training runs or start a new training session.
          </p>
        </Col>
      </Row>

      <Row>
        <Col md={8}>
          <Card>
            <Card.Header>
              <h4>Previous Runs</h4>
            </Card.Header>
            <Card.Body>
              {loading ? (
                <p>Loading runs...</p>
              ) : error ? (
                <p className="text-danger">{error}</p>
              ) : runs.length === 0 ? (
                <p>No previous runs found. Start a new training session to see results here.</p>
              ) : (
                <ListGroup variant="flush">
                  {runs.map((run) => (
                    <ListGroup.Item key={run.id} className="run-item">
                      <Row>
                        <Col md={9}>
                          <h5>Run {run.id}</h5>
                          <p className="mb-1">
                            <strong>Timestamp:</strong> {formatTimestamp(run.timestamp)}
                          </p>
                          <p className="mb-1">
                            <strong>Rounds:</strong> {run.rounds}
                          </p>
                          <p className="mb-0">
                            <strong>Ensemble Accuracy:</strong>{' '}
                            <span className="text-success">{(run.ensemble_accuracy * 100).toFixed(2)}%</span>
                          </p>
                        </Col>
                        <Col md={3} className="d-flex align-items-center justify-content-end">
                          <Button
                            as={Link}
                            to={`/run/${run.id}`}
                            variant="primary"
                          >
                            View Details
                          </Button>
                        </Col>
                      </Row>
                    </ListGroup.Item>
                  ))}
                </ListGroup>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col md={4}>
          <Card className="mb-4">
            <Card.Header>
              <h4>Quick Actions</h4>
            </Card.Header>
            <Card.Body>
              <Button
                as={Link}
                to="/train"
                variant="success"
                size="lg"
                className="w-100 mb-3"
              >
                Start New Training
              </Button>
              <Button
                onClick={() => window.location.reload()}
                variant="outline-secondary"
                className="w-100"
              >
                Refresh Dashboard
              </Button>
            </Card.Body>
          </Card>

          <Card>
            <Card.Header>
              <h4>System Info</h4>
            </Card.Header>
            <Card.Body>
              <p className="mb-2">
                <strong>Models:</strong> Random Forest, Gradient Boosting, Neural Network
              </p>
              <p className="mb-2">
                <strong>Default Configuration:</strong>
              </p>
              <ul>
                <li>Rounds: 10</li>
                <li>Clients: 3</li>
                <li>Models: 3</li>
              </ul>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard; 