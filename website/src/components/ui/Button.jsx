import React from 'react';
import { motion } from 'framer-motion';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  disabled = false,
  onClick,
  type = 'button',
  ...props 
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-miami-green-500 hover:bg-miami-green-600 text-white focus:ring-miami-green-500',
    secondary: 'bg-miami-orange-500 hover:bg-miami-orange-600 text-white focus:ring-miami-orange-500',
    outline: 'border-2 border-miami-green-500 text-miami-green-600 hover:bg-miami-green-500 hover:text-white focus:ring-miami-green-500',
    ghost: 'text-miami-green-600 hover:bg-miami-green-50 dark:hover:bg-miami-green-900/20 focus:ring-miami-green-500'
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-6 py-2.5 text-base',
    lg: 'px-8 py-3 text-lg'
  };
  
  const buttonClasses = `${baseClasses} ${variants[variant]} ${sizes[size]} ${className}`;
  
  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      className={buttonClasses}
      onClick={onClick}
      disabled={disabled}
      type={type}
      {...props}
    >
      {children}
    </motion.button>
  );
};

export default Button;