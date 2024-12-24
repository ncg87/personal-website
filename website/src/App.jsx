import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import AboutMe from './components/AboutMe';
import Footer from './components/Footer';
import HomePage from './components/HomePage';

const App = () => {
    return (
        <Router>
            <Header />
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/about" element={<AboutMe />} />
            </Routes>
            <Footer />
        </Router>
    );
};

export default App;
