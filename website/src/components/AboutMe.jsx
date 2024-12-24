import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';

const AboutMe = () => {
    return (
        <Box
            id="about"
            sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '70vh', // Ensures it spans most of the page
                padding: 3,
                backgroundColor: '#1e1e1e', // Matches your site's theme
            }}
        >
            <Paper
                elevation={3}
                sx={{
                    padding: 4,
                    maxWidth: '800px',
                    backgroundColor: '#2c3e50',
                    color: '#ecf0f1',
                    borderRadius: '8px',
                }}
            >
                <Typography
                    variant="h4"
                    component="h2"
                    sx={{
                        marginBottom: 2,
                        color: '#1abc9c', // Highlighted heading color
                        textAlign: 'center',
                    }}
                >
                    About Me
                </Typography>
                <Typography
                    variant="body1"
                    sx={{
                        lineHeight: 1.6,
                        textAlign: 'justify',
                    }}
                >
                    I'm a passionate and driven student at the University of Miami,
                    pursuing a triple major in Computer Science, Mathematics, and
                    Data Science & AI. With a strong academic background (GPA: 3.9)
                    and hands-on experience in machine learning and software
                    development, I'm dedicated to pushing the boundaries of
                    technology and innovation. My research work and projects span
                    from advanced machine learning applications to quantitative
                    analysis and mobile app development.
                </Typography>
            </Paper>
        </Box>
    );
};

export default AboutMe;
