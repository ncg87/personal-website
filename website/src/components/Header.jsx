import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';

const Header = () => {
    return (
        <>
            <AppBar
                position="static"
                sx={{
                    backgroundColor: 'rgba(0, 80, 48, 0.9)', // University of Miami green with transparency
                }}
            >
                <Toolbar>
                    {/* Main Title */}
                    <Typography
                        sx={{
                            flexGrow: 1,
                            fontSize: 'clamp(1.5rem, 4vw, 2.5rem)', // Dynamically scales between 1.5rem and 2.5rem
                            fontWeight: 'bold', // Make it bold for emphasis
                        }}
                    >
                        Nickolas (Nicko) Goodis
                    </Typography>

                    {/* Navigation Buttons */}
                    <Box
                        sx={{
                            display: 'flex',
                            gap: '5px', // Space between buttons
                        }}
                    >
                        <Button
                            color="inherit"
                            
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: 'clamp(0.5rem, 2vw, 1rem)', // Dynamically scales between 0.8rem and 1rem
                            }}
                            href="/"
                        >
                            Home
                        </Button>
                        <Button
                            color="inherit"
                            
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: 'clamp(0.5rem, 2vw, 1rem)', // Dynamically scales between 0.8rem and 1rem
                            }}
                            href="/about"
                        >
                            About Me
                        </Button>
                        <Button
                            color="inherit"
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: 'clamp(0.5rem, 2vw, 1rem)', // Dynamically scales between 0.8rem and 1rem
                            }}
                            href="/posts"
                        >
                            Posts
                        </Button>
                        <Button
                            color="inherit"
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: 'clamp(0.5rem, 2vw, 1rem)', // Dynamically scales between 0.8rem and 1rem
                            }}
                            href="/projects"
                        >
                            Projects
                        </Button>
                        <Button
                            color="inherit"
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: 'clamp(0.5rem, 2vw, 1rem)', // Dynamically scales between 0.8rem and 1rem
                            }}
                            href="/contact"
                        >
                            Contact
                        </Button>
                    </Box>
                </Toolbar>
            </AppBar>
        </>
    );
};

export default Header;
