import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Row, Col, Card, Button, Table, Badge } from 'react-bootstrap';
import axios from 'axios';

const RunDetails = () => {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [plotUrls, setPlotUrls] = useState({
    accuracy: null,
    weights: null,
  });

  useEffect(() => {
    const fetchRunDetails = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/run/${runId}`);
        setRunData(response.data);
        
        // Fetch plot URLs
        const accuracyResponse = await axios.get(`/api/plot/accuracy/${runId}`);
        const weightsResponse = await axios.get(`/api/plot/weights/${runId}`);
        
        setPlotUrls({
          accuracy: accuracyResponse.data.url,
          weights: weightsResponse.data.url,
        });
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching run details:', err);
        setError('Failed to fetch run details. Please try again later.');
        setLoading(false);
      }
    };

    fetchRunDetails();
  }, [runId]);

  const handleDownloadWeights = () => {
    window.open(`/api/download/weights/${runId}`, '_blank');
  };

  const handleDownloadModel = (modelIdx) => {
    window.open(`/api/download/model/${modelIdx}/${runId}`, '_blank');
  };

  const getLastRoundData = () => {
    if (!runData || !runData.results || runData.results.length === 0) {
      return null;
    }
    return runData.results[runData.results.length - 1];
  };

  const lastRound = getLastRoundData();

  return (
    <div>
      <Row className="mb-4">
        <Col>
          <h2>Run Details: {runId}</h2>
          <Button as={Link} to="/" variant="outline-secondary" className="me-2">
            Back to Dashboard
          </Button>
        </Col>
      </Row>

      {loading ? (
        <p>Loading run details...</p>
      ) : error ? (
        <p className="text-danger">{error}</p>
      ) : !runData ? (
        <p>No data available for this run.</p>
      ) : (
        <>
          <Row>
            <Col md={8}>
              <Card className="mb-4">
                <Card.Header>
                  <h4>Performance Over Time</h4>
                </Card.Header>
                <Card.Body>
                  {plotUrls.accuracy ? (
                    <img 
                      src={plotUrls.accuracy} 
                      alt="Accuracy Plot" 
                      className="img-fluid rounded"
                    />
                  ) : (
                    <p>Accuracy plot not available</p>
                  )}
                </Card.Body>
              </Card>

              <Card className="mb-4">
                <Card.Header>
                  <h4>Ensemble Weights</h4>
                </Card.Header>
                <Card.Body>
                  {plotUrls.weights ? (
                    <img 
                      src={plotUrls.weights} 
                      alt="Weights Plot" 
                      className="img-fluid rounded"
                    />
                  ) : (
                    <p>Weights plot not available</p>
                  )}
                </Card.Body>
              </Card>
            </Col>

            <Col md={4}>
              <Card className="mb-4">
                <Card.Header>
                  <h4>Final Metrics</h4>
                </Card.Header>
                <Card.Body>
                  {lastRound ? (
                    <Table striped bordered hover>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Accuracy</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Random Forest</td>
                          <td>{lastRound.model_0 ? (lastRound.model_0.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                        </tr>
                        <tr>
                          <td>Gradient Boosting</td>
                          <td>{lastRound.model_1 ? (lastRound.model_1.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                        </tr>
                        <tr>
                          <td>Neural Network</td>
                          <td>{lastRound.model_2 ? (lastRound.model_2.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                        </tr>
                        <tr className="table-primary">
                          <td><strong>Ensemble</strong></td>
                          <td><strong>{lastRound.ensemble ? (lastRound.ensemble.accuracy * 100).toFixed(2) + '%' : 'N/A'}</strong></td>
                        </tr>
                      </tbody>
                    </Table>
                  ) : (
                    <p>No metrics available</p>
                  )}
                </Card.Body>
              </Card>

              <Card className="mb-4">
                <Card.Header>
                  <h4>Final Ensemble Weights</h4>
                </Card.Header>
                <Card.Body>
                  {runData.weights ? (
                    <>
                      <Table striped bordered hover>
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>Random Forest</td>
                            <td>{runData.weights[0].toFixed(4)}</td>
                          </tr>
                          <tr>
                            <td>Gradient Boosting</td>
                            <td>{runData.weights[1].toFixed(4)}</td>
                          </tr>
                          <tr>
                            <td>Neural Network</td>
                            <td>{runData.weights[2].toFixed(4)}</td>
                          </tr>
                        </tbody>
                      </Table>
                      <Button 
                        variant="outline-primary" 
                        onClick={handleDownloadWeights}
                        className="w-100"
                      >
                        Download Weights
                      </Button>
                    </>
                  ) : (
                    <p>No weights available</p>
                  )}
                </Card.Body>
              </Card>

              <Card>
                <Card.Header>
                  <h4>Download Models</h4>
                </Card.Header>
                <Card.Body>
                  <div className="d-grid gap-2">
                    <Button 
                      variant="outline-info" 
                      onClick={() => handleDownloadModel(0)}
                    >
                      Download Random Forest
                    </Button>
                    <Button 
                      variant="outline-warning" 
                      onClick={() => handleDownloadModel(1)}
                    >
                      Download Gradient Boosting
                    </Button>
                    <Button 
                      variant="outline-secondary" 
                      onClick={() => handleDownloadModel(2)}
                    >
                      Download Neural Network
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          <Row className="mt-4">
            <Col>
              <Card>
                <Card.Header>
                  <h4>Detailed Round-by-Round Results</h4>
                </Card.Header>
                <Card.Body>
                  <Table responsive striped bordered hover>
                    <thead>
                      <tr>
                        <th>Round</th>
                        <th>Random Forest Accuracy</th>
                        <th>Gradient Boosting Accuracy</th>
                        <th>Neural Network Accuracy</th>
                        <th>Ensemble Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runData.results.map((round, index) => (
                        <tr key={index}>
                          <td>{index + 1}</td>
                          <td>{round.model_0 ? (round.model_0.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                          <td>{round.model_1 ? (round.model_1.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                          <td>{round.model_2 ? (round.model_2.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                          <td className="table-primary">
                            <strong>
                              {round.ensemble ? (round.ensemble.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                            </strong>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </>
      )}
    </div>
  );
};

export default RunDetails; 