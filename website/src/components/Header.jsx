import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';

const Header = () => {
    const [anchorEl, setAnchorEl] = React.useState(null);

    // Handle dropdown menu
    const handleMenuOpen = (event) => {
        setAnchorEl(event.currentTarget);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
    };

    return (
        <>
            <AppBar
                position="static"
                sx={{
                    backgroundColor: 'rgba(0, 80, 48, 0.9)', // University of Miami green with transparency
                }}
            >
                <Toolbar
                    sx={{
                        flexWrap: 'wrap', // Allow wrapping of content
                        minHeight: { xs: '56px', sm: '64px' }, // Smaller height for mobile devices
                        padding: '0 10px', // Adjust padding for smaller screens
                    }}
                >
                    {/* Main Title */}
                    <Typography
                        sx={{
                            flexGrow: 1,
                            fontSize: { xs: '1.2rem', sm: 'clamp(1.5rem, 4vw, 2.5rem)' }, // Smaller font size for mobile
                            fontWeight: 'bold', // Bold for emphasis
                        }}
                    >
                        Nickolas (Nicko) Goodis
                    </Typography>

                    {/* Navigation Buttons */}
                    <Box
                        sx={{
                            display: 'flex',
                            gap: '10px', // Increased spacing between buttons
                            flexWrap: 'wrap', // Allow buttons to wrap on smaller screens
                        }}
                    >
                        <Button
                            color="inherit"
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 2vw, 1rem)' }, // Smaller font size for mobile
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
                                fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 2vw, 1rem)' },
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
                                fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 2vw, 1rem)' },
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
                                fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 2vw, 1rem)' },
                            }}
                            href="/posts"
                        >
                            Posts
                        </Button>

                        {/* Contact Dropdown */}
                        <Button
                            color="inherit"
                            onClick={handleMenuOpen} // Opens the menu on click
                            sx={{
                                alignItems: 'center',
                                textAlign: 'center',
                                fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 2vw, 1rem)' },
                                backgroundColor: '#FF5722', // Orange background
                                borderRadius: '20px', // Rounded corners
                                color: 'white', // White text color
                                padding: { xs: '5px 10px', sm: '5px 15px' }, // Adjust padding for smaller screens
                                '&:hover': {
                                    backgroundColor: '#E64A19', // Darker orange on hover
                                },
                            }}
                        >
                            Contact
                        </Button>
                        <Menu
                            anchorEl={anchorEl}
                            open={Boolean(anchorEl)}
                            onClose={handleMenuClose} // Closes menu on click outside or on an item
                            sx={{
                                mt: '5px', // Adjust dropdown positioning
                                '& .MuiPaper-root': {
                                    backgroundColor: 'rgba(0, 80, 48, 0.95)', // Dropdown background color
                                    color: 'white', // Text color
                                    borderRadius: '8px', // Rounded corners
                                },
                            }}
                        >
                            <MenuItem
                                onClick={() => {
                                    handleMenuClose();
                                    window.location.href = 'mailto:ncg87@miami.edu';
                                }}
                                sx={{
                                    fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 1.5vw, 1rem)' }, // Responsive font size
                                }}
                            >
                                Email
                            </MenuItem>
                            <MenuItem
                                onClick={() => {
                                    handleMenuClose();
                                    window.open('/resume.pdf', '_blank');
                                }}
                                sx={{
                                    fontSize: { xs: '0.7rem', sm: 'clamp(0.8rem, 1.5vw, 1rem)' }, // Responsive font size
                                }}
                            >
                                Resume
                            </MenuItem>
                        </Menu>
                    </Box>
                </Toolbar>
            </AppBar>
        </>
    );
};

export default Header;
