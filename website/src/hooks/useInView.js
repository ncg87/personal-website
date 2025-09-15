import { useState, useEffect, useRef } from 'react';

const useInView = (options = {}) => {
  const [isInView, setIsInView] = useState(false);
  const [hasBeenInView, setHasBeenInView] = useState(false);
  const ref = useRef();

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          if (!hasBeenInView) {
            setHasBeenInView(true);
          }
        } else {
          setIsInView(false);
        }
      },
      {
        threshold: 0.1,
        rootMargin: '-50px 0px',
        ...options,
      }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [hasBeenInView, options]);

  return { ref, isInView, hasBeenInView };
};

export default useInView;