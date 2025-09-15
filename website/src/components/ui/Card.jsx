import React from 'react';
import { motion } from 'framer-motion';

const Card = ({ 
  children, 
  className = '', 
  hover = true,
  padding = 'md',
  ...props 
}) => {
  const baseClasses = 'bg-white dark:bg-miami-neutral-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm';
  const hoverClasses = hover ? 'hover:shadow-lg transition-shadow duration-300' : '';
  
  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };
  
  const cardClasses = `${baseClasses} ${hoverClasses} ${paddingClasses[padding]} ${className}`;
  
  if (hover) {
    return (
      <motion.div
        whileHover={{ y: -2 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className={cardClasses}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
  
  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  );
};

export default Card;