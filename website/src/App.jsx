import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import AboutMe from './components/AboutMe';
import Footer from './components/Footer';

const App = () => {
    return (
        <Router>
            <Header />
            <Routes>
                <Route path="/" element={<AboutMe />} />
                <Route path="/about" element={<AboutMe />} />
            </Routes>
            <Footer />
        </Router>
    );
};

export default App;
