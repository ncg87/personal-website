import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const PlaceholderPage = ({ title }) => (
    <Box
        sx={{
            display: 'flex',
            flexDirection: 'column', // Ensures content stacks vertically
            justifyContent: 'center', // Centers content vertically
            alignItems: 'center', // Centers content horizontally
            minHeight: 'calc(100vh - 128px)', // Ensures full viewport height minus header/footer
            width: '100%', // Takes the full width of the container
            padding: '20px',
            textAlign: 'center', // Centers text horizontally
            overflow: 'hidden', // Prevents overflow of content
        }}
    >
        <Typography
            variant="h1"
            sx={{
                fontSize: 'clamp(2rem, 5vw, 4rem)', // Dynamically scales font size
                fontWeight: 'bold',
                color: 'white',
            }}
        >
            {title} Page Coming Soon
        </Typography>
    </Box>
);

export default PlaceholderPage;
