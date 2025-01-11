import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import { Link } from 'react-router-dom';

const projects = [
    {
        title: "CryptoFlows.ai",
        description:
            "Current work in progress, working on the backend, track every individual transaction that goes through a blochain network.",
        github: "https://github.com/ncg87/cryptoflows",
        website: "https://cryptoflows.ai",
        postLink: "/posts/cryptoflows-ai",
    },
    {
        title: "Bitcoin Transaction Network Visualization",
        description:
            "Allows users to explore the relationships between Bitcoin transactions, addresses, and outputs in an intuitive three-dimensional space. Built on top of my CryptoFlows.ai backend.",
        github: "https://github.com/ncg87/bitcoin-tx-graph",
        website: "https://bitcoin-tx-graph.vercel.app/",
        postLink: "/posts/bitcoin-transaction-network-visualization",
    },
    {
        title: "DEX Tracker",
        description:
            "Built a DEX tracker that every transaction on a couple of different DEXs down to the grainular level an stores it in a database.",
        github: "https://github.com/ncg87/dex_tracker_system",
        postLink: "/posts/dex-tracker",
    },
    {
        title: "Donation Blockchain Portal",
        description:
            "Built a decentralized app for secure ETH transactions and donor tier tracking using Solidity and React.",
        github: "https://github.com/ncg87/donation-app",
        website: "https://donation-app-roan.vercel.app/",
        postLink: "/posts/donation-blockchain-portal",
    },
    {
        title: "Automated AI Researcher",
        description:
            "Developed an interactive CLI tool leveraging LLM APIs to automate research analysis of arXiv papers.",
        github: "https://github.com/ncg87/Research-Assistant",
        postLink: "/posts/automated-ai-researcher",
    },
    {
        title: "Personal Website",
        description:
            "Created a personal website to showcase my projects and blog posts.",
        github: "https://github.com/ncg87/personal-website",
        website: "https://nickolasgoodis.com",
        postLink: "/posts/personal-website",
    },
    {
        title: "Machine Learning Deployment Platform",
        description:
            "Architected a Flask-based website showcasing multiple ML models, including multilingual translation and image captioning.",
        github: "https://github.com/ncg87/projects-website",
        website: "https://nickogoodis.com",
        postLink: "/posts/machine-learning-deployment-platform",
    },
    {
        title: "Face Detector Model",
        description:
            "Created a YOLO model that detects faces in live video.",
        github: "https://github.com/ncg87/YOLO",
        postLink: "/posts/face-detector",
    },
    {
        title: "Multilingual Translation Model",
        description:
            "Utilized a transfer learning approach to create a model that takes an one of 6 languages as input and outputs the translation to one of 6 languages.",
        github: "https://github.com/ncg87/machine-translation",
        postLink: "/posts/translation-model",
    },
    {
        title: "Image Captioning Model",
        description:
            "Created a model that takes an image as input and outputs a caption of the image traid on the Flicker8k dataset.",
        github: "https://github.com/ncg87/image-captioning",
        postLink: "/posts/image-captioning-model",
    },
    {
        title: "Pairs Trading Statistical Arbitrage Optimizer",
        description:
            "Engineered a predictive trading model using PyTorch and Alpaca API, achieving significant backtested returns.",
        github: "https://github.com/ncg87/pairs-trading-bot",
        postLink: "/posts/pairs-trading-statistical-arbitrage-optimizer",
    },
    {
        title: "Python Tic-Tac-Toe Game",
        description:
            "Created a simple Tic-Tac-Toe game in Python, allowing two players to play against each other or to play against a bot of varying difficulties. ",
        github: "https://github.com/ncg87/tic-tac-toe",
        postLink: "/posts/python-tic-tac-toe-game",
    }

];

const ProjectsPage = () => {
    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '100vh',
                padding: '20px',
                backgroundColor: 'rgba(30, 30, 30, 0.5)',
            }}
        >
            <Typography
                variant="h3"
                sx={{
                    marginBottom: '20px',
                    color: '#ffffff', // White color for heading
                    textAlign: 'center',
                }}
            >
                Projects
            </Typography>
            <Grid container spacing={3} justifyContent="center">
                {projects.map((project, index) => (
                    <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card
                            sx={{
                                backgroundColor: 'rgba(0, 60, 36, 1)',
                                color: '#ecf0f1',
                                borderRadius: '8px',
                                boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                                height: '100%',
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'center',
                            }}
                        >
                            <CardContent>
                                <Typography
                                    variant="h5"
                                    component={Link} // Use Link component for navigation
                                    to={project.postLink} // Dynamic link to the post
                                    sx={{
                                        color: '#ffffff', // White for the title
                                        textDecoration: 'none', // Remove underline
                                        '&:hover': { color: '#dcdcdc' }, // Lighter white on hover
                                        marginBottom: '10px',
                                        display: 'flex',
                                        justifyContent: 'center',
                                    }}
                                >
                                    {project.title}
                                </Typography>
                                <Typography
                                    variant="body2"
                                    sx={{
                                        lineHeight: 1.6,
                                        color: '#dcdcdc', // Off-white for description
                                    }}
                                >
                                    {project.description}
                                </Typography>
                            </CardContent>
                            <Box
                                sx={{
                                    display: 'flex',
                                    justifyContent: 'center',
                                    padding: '10px',
                                }}
                            >
                                <Button
                                    href={project.github}
                                    target="_blank"
                                    sx={{
                                        marginRight: '5px',
                                        color: '#333333', // Dark text on light gray
                                        backgroundColor: '#dcdcdc', // Light gray buttons
                                        '&:hover': { backgroundColor: '#bfbfbf' }, // Slightly darker gray on hover
                                    }}
                                >
                                    GitHub
                                </Button>
                                {project.website && (
                                    <Button
                                        href={project.website}
                                        target="_blank"
                                        sx={{
                                            marginLeft: '5px',
                                            color: '#333333', // Dark text on light gray
                                            backgroundColor: '#dcdcdc', // Light gray buttons
                                            '&:hover': { backgroundColor: '#bfbfbf' }, // Slightly darker gray on hover
                                        }}
                                    >
                                        Website
                                    </Button>
                                )}
                            </Box>
                        </Card>
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
};

export default ProjectsPage;
