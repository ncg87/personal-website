import React from 'react';
import { motion } from 'framer-motion';
import useReducedMotion from '../../hooks/useReducedMotion';

const LoadingSpinner = ({ 
  size = 'w-8 h-8', 
  color = 'text-miami-green-600', 
  className = '',
  variant = 'spin'
}) => {
  const shouldReduceMotion = useReducedMotion();

  const spinVariants = {
    animate: {
      rotate: shouldReduceMotion ? 0 : 360,
      transition: {
        duration: shouldReduceMotion ? 0 : 1,
        repeat: shouldReduceMotion ? 0 : Infinity,
        ease: 'linear'
      }
    }
  };

  const pulseVariants = {
    animate: {
      scale: shouldReduceMotion ? 1 : [1, 1.2, 1],
      opacity: shouldReduceMotion ? 1 : [1, 0.8, 1],
      transition: {
        duration: shouldReduceMotion ? 0 : 1.5,
        repeat: shouldReduceMotion ? 0 : Infinity,
        ease: 'easeInOut'
      }
    }
  };

  const bounceVariants = {
    animate: {
      y: shouldReduceMotion ? 0 : [0, -10, 0],
      transition: {
        duration: shouldReduceMotion ? 0 : 0.6,
        repeat: shouldReduceMotion ? 0 : Infinity,
        ease: 'easeInOut'
      }
    }
  };

  const getVariantAnimation = () => {
    switch (variant) {
      case 'pulse':
        return pulseVariants;
      case 'bounce':
        return bounceVariants;
      default:
        return spinVariants;
    }
  };

  if (variant === 'dots') {
    return (
      <div className={`flex space-x-1 ${className}`}>
        {[0, 1, 2].map((index) => (
          <motion.div
            key={index}
            className={`w-2 h-2 ${color.replace('text-', 'bg-')} rounded-full`}
            animate={shouldReduceMotion ? {} : {
              y: [0, -8, 0],
              transition: {
                duration: 0.6,
                repeat: Infinity,
                delay: index * 0.1,
                ease: 'easeInOut'
              }
            }}
          />
        ))}
      </div>
    );
  }

  if (variant === 'bars') {
    return (
      <div className={`flex space-x-1 ${className}`}>
        {[0, 1, 2, 3].map((index) => (
          <motion.div
            key={index}
            className={`w-1 h-6 ${color.replace('text-', 'bg-')} rounded-full`}
            animate={shouldReduceMotion ? {} : {
              scaleY: [1, 0.3, 1],
              transition: {
                duration: 0.8,
                repeat: Infinity,
                delay: index * 0.1,
                ease: 'easeInOut'
              }
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <motion.div
      className={`${size} ${className}`}
      variants={getVariantAnimation()}
      animate="animate"
    >
      <svg
        className={`${size} ${color} animate-spin`}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    </motion.div>
  );
};

// Pre-built loading states
export const PageLoader = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="text-center">
      <LoadingSpinner size="w-12 h-12" className="mx-auto mb-4" />
      <p className="text-gray-600 dark:text-gray-300">Loading...</p>
    </div>
  </div>
);

export const ButtonLoader = () => (
  <LoadingSpinner size="w-4 h-4" />
);

export const InlineLoader = ({ text = 'Loading' }) => (
  <div className="flex items-center gap-2">
    <LoadingSpinner size="w-4 h-4" />
    <span className="text-sm text-gray-600 dark:text-gray-300">{text}...</span>
  </div>
);

export default LoadingSpinner;