import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Button from './ui/Button';
import useReducedMotion from '../hooks/useReducedMotion';

const ModernHeader = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const shouldReduceMotion = useReducedMotion();

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'About', href: '/about' },
    { name: 'Projects', href: '/projects' },
    { name: 'Posts', href: '/posts' },
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
    import('../utils/analytics').then(({ trackResumeDownload }) => {
      trackResumeDownload('Header Download');
    });
    
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
            {navigation.map((item, index) => (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, y: shouldReduceMotion ? 0 : -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  delay: shouldReduceMotion ? 0 : index * 0.1,
                  duration: shouldReduceMotion ? 0 : 0.5
                }}
                whileHover={shouldReduceMotion ? {} : { 
                  scale: 1.05,
                  transition: { duration: 0.2 }
                }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
              >
                <Link
                  to={item.href}
                  className={`relative px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 overflow-hidden group ${
                    isActive(item.href)
                      ? 'bg-miami-green-600 text-white'
                      : 'text-miami-green-100 hover:text-white'
                  }`}
                >
                  {/* Animated background on hover */}
                  {!isActive(item.href) && (
                    <motion.div
                      className="absolute inset-0 bg-miami-green-600 origin-left"
                      initial={{ scaleX: 0 }}
                      whileHover={{ scaleX: 1 }}
                      transition={{ 
                        duration: shouldReduceMotion ? 0 : 0.4,
                        ease: [0.25, 0.46, 0.45, 0.94]
                      }}
                    />
                  )}
                  
                  {/* Glowing effect */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-miami-green-400 to-miami-orange-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: shouldReduceMotion ? 0 : 0.25 }}
                    transition={{ duration: shouldReduceMotion ? 0 : 0.3 }}
                  />
                  
                  {/* Shimmer effect */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 -skew-x-12"
                    initial={{ x: '-100%', opacity: 0 }}
                    whileHover={{ x: '100%', opacity: 1 }}
                    transition={{ 
                      duration: shouldReduceMotion ? 0 : 0.6,
                      ease: 'easeInOut',
                      delay: shouldReduceMotion ? 0 : 0.1
                    }}
                  />
                  
                  <span className="relative z-10">{item.name}</span>
                </Link>
              </motion.div>
            ))}
            
            {/* Resume Download Button */}
            <motion.div
              whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
              whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
              className="ml-4"
            >
              <Button
                onClick={handleDownloadResume}
                variant="secondary"
                size="sm"
                className="group relative overflow-hidden"
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-miami-orange-400 to-miami-orange-500 opacity-0 group-hover:opacity-100"
                  initial={{ opacity: 0 }}
                  whileHover={{ opacity: shouldReduceMotion ? 0 : 1 }}
                  transition={{ duration: shouldReduceMotion ? 0 : 0.3 }}
                />
                <Download className="w-4 h-4 mr-2 relative z-10 group-hover:animate-bounce" />
                <span className="relative z-10">Resume PDF</span>
              </Button>
            </motion.div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <motion.button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-miami-green-100 hover:text-white hover:bg-miami-green-600 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white transition-colors duration-200"
              aria-expanded="false"
              whileHover={shouldReduceMotion ? {} : { scale: 1.1 }}
              whileTap={shouldReduceMotion ? {} : { scale: 0.9 }}
            >
              <span className="sr-only">Open main menu</span>
              <motion.div
                animate={{ rotate: isMenuOpen ? 180 : 0 }}
                transition={{ duration: shouldReduceMotion ? 0 : 0.3 }}
              >
                {isMenuOpen ? (
                  <X className="block h-6 w-6" aria-hidden="true" />
                ) : (
                  <Menu className="block h-6 w-6" aria-hidden="true" />
                )}
              </motion.div>
            </motion.button>
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
                  <motion.div
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <Button
                      onClick={() => {
                        handleDownloadResume();
                        setIsMenuOpen(false);
                      }}
                      variant="secondary"
                      size="sm"
                      className="w-full group"
                    >
                      <Download className="w-4 h-4 mr-2 group-hover:animate-bounce" />
                      Download Resume PDF
                    </Button>
                  </motion.div>
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