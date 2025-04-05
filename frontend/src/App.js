import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

import NavigationBar from './components/NavigationBar';
import Dashboard from './components/Dashboard';
import RunDetails from './components/RunDetails';
import TrainingControl from './components/TrainingControl';

function App() {
  return (
    <Router>
      <div className="App">
        <NavigationBar />
        <div className="container mt-4">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/run/:runId" element={<RunDetails />} />
            <Route path="/train" element={<TrainingControl />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App; 