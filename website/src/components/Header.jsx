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
                    backgroundColor: 'rgba(244, 115, 33, 0.8)', // Orange color with slight transparency
                }}
            >
                <Toolbar>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Nickolas (Nicko) Goodis
                    </Typography>
                    <Button color="inherit" href="/">Home</Button>
                    <Button color="inherit" href="/about">About Me</Button>
                    <Button color="inherit" href="/projects">Projects</Button>
                </Toolbar>
            </AppBar>
        </>
    );
};

export default Header;
