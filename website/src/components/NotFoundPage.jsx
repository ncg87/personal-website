import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Home, ArrowLeft, Search, FileX } from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import SEO from './SEO';
import PageTransition from './ui/PageTransition';
import useReducedMotion from '../hooks/useReducedMotion';

const NotFoundPage = () => {
  const shouldReduceMotion = useReducedMotion();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: shouldReduceMotion ? 0 : 0.2,
        delayChildren: shouldReduceMotion ? 0 : 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { 
      opacity: 0, 
      y: shouldReduceMotion ? 0 : 30 
    },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: shouldReduceMotion ? 0 : 0.6,
        ease: 'easeOut'
      }
    }
  };

  const floatingVariants = {
    animate: shouldReduceMotion ? {} : {
      y: [-10, 10, -10],
      rotate: [-5, 5, -5],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <>
      <SEO 
        title="404 - Page Not Found | Nickolas Goodis"
        description="The page you're looking for doesn't exist. Return to the homepage or explore other sections of Nickolas Goodis's portfolio."
        keywords="404, page not found, error"
        url="https://nickogoodis.com/404"
      />
      
      <PageTransition>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 dark:from-miami-neutral-900 dark:via-miami-neutral-900 dark:to-gray-900 flex items-center justify-center px-4 sm:px-6 lg:px-8">
          
          {/* Background Pattern */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute inset-0 opacity-5">
              <div 
                className="absolute inset-0"
                style={{
                  backgroundImage: `radial-gradient(circle at 1px 1px, rgba(5, 80, 48, 0.3) 1px, transparent 0)`,
                  backgroundSize: '50px 50px'
                }}
              />
            </div>
          </div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="relative max-w-2xl mx-auto text-center"
          >
            
            {/* 404 Icon */}
            <motion.div
              variants={floatingVariants}
              animate="animate"
              className="mb-8"
            >
              <div className="relative">
                <motion.div
                  variants={itemVariants}
                  className="text-8xl sm:text-9xl font-bold text-miami-green-500/20 dark:text-miami-green-400/20"
                >
                  404
                </motion.div>
                <motion.div
                  variants={itemVariants}
                  className="absolute inset-0 flex items-center justify-center"
                >
                  <FileX className="w-16 h-16 sm:w-20 sm:h-20 text-miami-green-600 dark:text-miami-green-400" />
                </motion.div>
              </div>
            </motion.div>

            {/* Error Message */}
            <motion.div variants={itemVariants} className="mb-8">
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Page Not Found
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300 mb-2">
                Oops! The page you're looking for doesn't exist.
              </p>
              <p className="text-gray-500 dark:text-gray-400">
                It might have been moved, deleted, or you entered the wrong URL.
              </p>
            </motion.div>

            {/* Action Cards */}
            <motion.div variants={itemVariants} className="mb-8">
              <Card className="p-6" padding="none">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  What would you like to do?
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  
                  {/* Go Home */}
                  <motion.div
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <Link to="/">
                      <Button 
                        variant="primary" 
                        size="lg" 
                        className="w-full group"
                      >
                        <Home className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                        Go Home
                      </Button>
                    </Link>
                  </motion.div>

                  {/* View Projects */}
                  <motion.div
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <Link to="/projects">
                      <Button 
                        variant="outline" 
                        size="lg" 
                        className="w-full group"
                      >
                        <Search className="w-5 h-5 mr-2 group-hover:rotate-12 transition-transform" />
                        View Projects
                      </Button>
                    </Link>
                  </motion.div>

                </div>
              </Card>
            </motion.div>

            {/* Additional Navigation */}
            <motion.div variants={itemVariants}>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                Or explore other sections:
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <Link 
                  to="/about" 
                  className="text-miami-green-600 hover:text-miami-green-700 dark:text-miami-green-400 dark:hover:text-miami-green-300 transition-colors"
                >
                  About Me
                </Link>
                <span className="text-gray-300 dark:text-gray-600">•</span>
                <Link 
                  to="/resume" 
                  className="text-miami-green-600 hover:text-miami-green-700 dark:text-miami-green-400 dark:hover:text-miami-green-300 transition-colors"
                >
                  Resume
                </Link>
                <span className="text-gray-300 dark:text-gray-600">•</span>
                <Link 
                  to="/contact" 
                  className="text-miami-green-600 hover:text-miami-green-700 dark:text-miami-green-400 dark:hover:text-miami-green-300 transition-colors"
                >
                  Contact
                </Link>
              </div>
            </motion.div>

            {/* Terminal Style Message */}
            <motion.div 
              variants={itemVariants}
              className="mt-8 p-4 bg-black/90 rounded-lg border border-miami-green-500/30 font-mono text-sm"
            >
              <div className="text-miami-green-400">
                <span className="text-miami-green-600">$</span> curl -I {window.location.href}
                <br />
                <span className="text-red-400">HTTP/1.1 404 Not Found</span>
                <br />
                <span className="text-miami-green-600">$</span> echo "Redirecting to home..."
                <br />
                <span className="text-gray-400">// Consider visiting /home instead</span>
              </div>
            </motion.div>

          </motion.div>
        </div>
      </PageTransition>
    </>
  );
};

export default NotFoundPage;