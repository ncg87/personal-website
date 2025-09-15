import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sun, Moon, Monitor, ChevronDown } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import useReducedMotion from '../../hooks/useReducedMotion';

const ThemeToggle = ({ variant = 'button', className = '' }) => {
  const { theme, toggleTheme, setLightTheme, setDarkTheme, setSystemTheme, isDark } = useTheme();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  const themeOptions = [
    { id: 'light', name: 'Light', icon: Sun, action: setLightTheme },
    { id: 'dark', name: 'Dark', icon: Moon, action: setDarkTheme },
    { id: 'system', name: 'System', icon: Monitor, action: setSystemTheme }
  ];

  const getCurrentThemeOption = () => {
    const savedTheme = localStorage.getItem('theme');
    if (!savedTheme) return themeOptions.find(opt => opt.id === 'system');
    return themeOptions.find(opt => opt.id === savedTheme) || themeOptions[0];
  };

  const currentOption = getCurrentThemeOption();

  if (variant === 'simple') {
    return (
      <motion.button
        onClick={toggleTheme}
        className={`p-2 rounded-lg text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors ${className}`}
        whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
        whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
        aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
      >
        <AnimatePresence mode="wait">
          {isDark ? (
            <motion.div
              key="sun"
              initial={{ opacity: 0, rotate: shouldReduceMotion ? 0 : -90 }}
              animate={{ opacity: 1, rotate: 0 }}
              exit={{ opacity: 0, rotate: shouldReduceMotion ? 0 : 90 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            >
              <Sun className="w-5 h-5" />
            </motion.div>
          ) : (
            <motion.div
              key="moon"
              initial={{ opacity: 0, rotate: shouldReduceMotion ? 0 : 90 }}
              animate={{ opacity: 1, rotate: 0 }}
              exit={{ opacity: 0, rotate: shouldReduceMotion ? 0 : -90 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            >
              <Moon className="w-5 h-5" />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>
    );
  }

  // Full dropdown version
  return (
    <div className={`relative ${className}`}>
      <motion.button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
        whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
        aria-label="Select theme"
        aria-expanded={isDropdownOpen}
      >
        <currentOption.icon className="w-4 h-4" />
        <span className="text-sm font-medium">{currentOption.name}</span>
        <motion.div
          animate={{ rotate: isDropdownOpen ? 180 : 0 }}
          transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
        >
          <ChevronDown className="w-4 h-4" />
        </motion.div>
      </motion.button>

      <AnimatePresence>
        {isDropdownOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-10"
              onClick={() => setIsDropdownOpen(false)}
            />
            
            {/* Dropdown */}
            <motion.div
              initial={{ opacity: 0, y: shouldReduceMotion ? 0 : -10, scale: shouldReduceMotion ? 1 : 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: shouldReduceMotion ? 0 : -10, scale: shouldReduceMotion ? 1 : 0.95 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
              className="absolute top-full right-0 mt-2 w-36 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-1 z-20"
            >
              {themeOptions.map((option) => {
                const Icon = option.icon;
                const isSelected = currentOption.id === option.id;
                
                return (
                  <motion.button
                    key={option.id}
                    onClick={() => {
                      option.action();
                      setIsDropdownOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                      isSelected
                        ? 'bg-miami-green-50 dark:bg-miami-green-900/20 text-miami-green-600 dark:text-miami-green-400'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }`}
                    whileHover={shouldReduceMotion ? {} : { x: 2 }}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{option.name}</span>
                    {isSelected && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="ml-auto w-2 h-2 bg-miami-green-500 rounded-full"
                      />
                    )}
                  </motion.button>
                );
              })}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ThemeToggle;