import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, X, Filter } from 'lucide-react';
import useReducedMotion from '../../hooks/useReducedMotion';

const SearchBar = ({ 
  onSearch, 
  onFilterChange, 
  availableFilters = [], 
  selectedFilters = [],
  placeholder = "Search...",
  className = ""
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    const delayedSearch = setTimeout(() => {
      onSearch(searchTerm);
    }, 300); // Debounce search

    return () => clearTimeout(delayedSearch);
  }, [searchTerm, onSearch]);

  const handleFilterToggle = (filter) => {
    const newFilters = selectedFilters.includes(filter)
      ? selectedFilters.filter(f => f !== filter)
      : [...selectedFilters, filter];
    onFilterChange(newFilters);
  };

  const clearSearch = () => {
    setSearchTerm('');
    onSearch('');
  };

  const clearAllFilters = () => {
    onFilterChange([]);
  };

  return (
    <div className={`relative ${className}`}>
      {/* Search Input */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="block w-full pl-10 pr-12 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-miami-green-500 focus:border-miami-green-500 bg-white dark:bg-miami-neutral-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 transition-colors"
          placeholder={placeholder}
        />
        
        <div className="absolute inset-y-0 right-0 flex items-center">
          {searchTerm && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={clearSearch}
              className="p-1 mx-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </motion.button>
          )}
          
          {availableFilters.length > 0 && (
            <motion.button
              onClick={() => setIsFilterOpen(!isFilterOpen)}
              className={`p-2 mr-1 rounded-md transition-colors ${
                selectedFilters.length > 0 || isFilterOpen
                  ? 'text-miami-green-600 bg-miami-green-50 dark:bg-miami-green-900/20'
                  : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'
              }`}
              whileHover={shouldReduceMotion ? {} : { scale: 1.05 }}
              whileTap={shouldReduceMotion ? {} : { scale: 0.95 }}
              aria-label="Toggle filters"
            >
              <Filter className="h-4 w-4" />
              {selectedFilters.length > 0 && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-1 -right-1 bg-miami-green-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center"
                >
                  {selectedFilters.length}
                </motion.span>
              )}
            </motion.button>
          )}
        </div>
      </div>

      {/* Filter Dropdown */}
      <AnimatePresence>
        {isFilterOpen && availableFilters.length > 0 && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-10"
              onClick={() => setIsFilterOpen(false)}
            />
            
            {/* Filter Panel */}
            <motion.div
              initial={{ opacity: 0, y: shouldReduceMotion ? 0 : -10, scale: shouldReduceMotion ? 1 : 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: shouldReduceMotion ? 0 : -10, scale: shouldReduceMotion ? 1 : 0.95 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
              className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-3 z-20"
            >
              <div className="px-3 mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Filter by tags</span>
                {selectedFilters.length > 0 && (
                  <button
                    onClick={clearAllFilters}
                    className="text-xs text-miami-green-600 hover:text-miami-green-700 dark:text-miami-green-400 dark:hover:text-miami-green-300"
                  >
                    Clear all
                  </button>
                )}
              </div>
              
              <div className="max-h-48 overflow-y-auto">
                {availableFilters.map((filter) => {
                  const isSelected = selectedFilters.includes(filter);
                  
                  return (
                    <motion.button
                      key={filter}
                      onClick={() => handleFilterToggle(filter)}
                      className={`w-full flex items-center justify-between px-3 py-2 text-sm transition-colors ${
                        isSelected
                          ? 'bg-miami-green-50 dark:bg-miami-green-900/20 text-miami-green-600 dark:text-miami-green-400'
                          : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                      whileHover={shouldReduceMotion ? {} : { x: 2 }}
                    >
                      <span>{filter}</span>
                      {isSelected && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="w-2 h-2 bg-miami-green-500 rounded-full"
                        />
                      )}
                    </motion.button>
                  );
                })}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Active Filters Display */}
      <AnimatePresence>
        {selectedFilters.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.2 }}
            className="mt-2 flex flex-wrap gap-2"
          >
            {selectedFilters.map((filter) => (
              <motion.span
                key={filter}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="inline-flex items-center gap-1 px-2 py-1 bg-miami-green-100 dark:bg-miami-green-900/30 text-miami-green-700 dark:text-miami-green-300 text-xs rounded-full"
              >
                {filter}
                <button
                  onClick={() => handleFilterToggle(filter)}
                  className="hover:text-miami-green-900 dark:hover:text-miami-green-100"
                  aria-label={`Remove ${filter} filter`}
                >
                  <X className="h-3 w-3" />
                </button>
              </motion.span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchBar;