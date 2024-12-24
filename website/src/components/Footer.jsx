import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';

const Footer = () => {
    return (
        <Box
            component="footer"
            sx={{
                width: '100vw', // Full width of the viewport
                 backgroundColor: 'rgba(244, 115, 33, 0.9)',
                color: '#fff',
                textAlign: 'center',
                padding: '20px 0',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 1,
            }}
        >
            {/* Social Icons */}
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
                <IconButton
                    href="https://github.com/ncg87"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ color: '#fff' }}
                >
                    <GitHubIcon fontSize="large" />
                </IconButton>
                <IconButton
                    href="https://www.linkedin.com/in/nickolas-charles-goodis-b82649258/"
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{ color: '#fff' }}
                >
                    <LinkedInIcon fontSize="large" />
                </IconButton>
            </Box>

            {/* Copyright Text */}
            <Typography variant="body2">
                © {new Date().getFullYear()} Nickolas Goodis. All rights reserved.
            </Typography>
        </Box>
    );
};

export default Footer;
