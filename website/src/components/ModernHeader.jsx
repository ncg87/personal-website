import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Button from './ui/Button';

const ModernHeader = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'About', href: '/about' },
    { name: 'Projects', href: '/projects' },
    { name: 'Resume', href: '/resume' },
    { name: 'Contact', href: '/contact' },
  ];

  const isActive = (href) => {
    if (href === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(href);
  };

  const handleDownloadResume = () => {
    // Track download event
    if (window.gtag) {
      window.gtag('event', 'resume_download_header', {
        event_category: 'engagement',
        event_label: 'Header Download'
      });
    }
    
    // Download PDF
    const link = document.createElement('a');
    link.href = '/resume.pdf';
    link.download = 'Nickolas_Goodis_Resume.pdf';
    link.click();
  };

  return (
    <header className="sticky top-0 z-50 bg-miami-green-500/95 backdrop-blur-sm border-b border-miami-green-600">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo/Name */}
          <Link 
            to="/" 
            className="flex items-center space-x-2 text-white font-bold text-lg sm:text-xl hover:text-miami-green-100 transition-colors"
          >
            <span>Nickolas Goodis</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => (
              <Link
                key={item.name}
                to={item.href}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.href)
                    ? 'bg-miami-green-600 text-white'
                    : 'text-miami-green-100 hover:bg-miami-green-600 hover:text-white'
                }`}
              >
                {item.name}
              </Link>
            ))}
            
            {/* Resume Download Button */}
            <Button
              onClick={handleDownloadResume}
              variant="secondary"
              size="sm"
              className="ml-4"
            >
              <Download className="w-4 h-4 mr-2" />
              Resume PDF
            </Button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-miami-green-100 hover:text-white hover:bg-miami-green-600 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
              aria-expanded="false"
            >
              <span className="sr-only">Open main menu</span>
              {isMenuOpen ? (
                <X className="block h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="block h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        <AnimatePresence>
          {isMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease: 'easeInOut' }}
              className="md:hidden overflow-hidden"
            >
              <div className="px-2 pt-2 pb-3 space-y-1 bg-miami-green-600 rounded-b-lg">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setIsMenuOpen(false)}
                    className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                      isActive(item.href)
                        ? 'bg-miami-green-700 text-white'
                        : 'text-miami-green-100 hover:bg-miami-green-700 hover:text-white'
                    }`}
                  >
                    {item.name}
                  </Link>
                ))}
                
                {/* Mobile Resume Download */}
                <div className="pt-2 mt-2 border-t border-miami-green-500">
                  <Button
                    onClick={() => {
                      handleDownloadResume();
                      setIsMenuOpen(false);
                    }}
                    variant="secondary"
                    size="sm"
                    className="w-full"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Resume PDF
                  </Button>
                </div>

                {/* Mobile Contact Links */}
                <div className="pt-2 mt-2 border-t border-miami-green-500">
                  <a
                    href="mailto:ncg87@miami.edu"
                    className="block px-3 py-2 rounded-md text-base font-medium text-miami-green-100 hover:bg-miami-green-700 hover:text-white transition-colors"
                  >
                    Email Me
                  </a>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>
    </header>
  );
};

export default ModernHeader;