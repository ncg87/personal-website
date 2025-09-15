import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    historyApiFallback: true,
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks
          react: ['react', 'react-dom'],
          router: ['react-router-dom'],
          animations: ['framer-motion'],
          icons: ['lucide-react'],
          
          // UI chunks  
          ui: [
            './src/components/ui/Button.jsx',
            './src/components/ui/Card.jsx',
            './src/components/ui/Badge.jsx',
            './src/components/ui/AnimatedSection.jsx',
            './src/components/ui/LoadingSkeleton.jsx',
            './src/components/ui/LoadingSpinner.jsx'
          ],
          
          // Page chunks
          pages: [
            './src/components/ModernAboutPage.jsx',
            './src/components/Resume.jsx',
            './src/components/ContactForm.jsx',
            './src/components/ModernProjectsPage.jsx',
            './src/components/ProjectCaseStudy.jsx'
          ]
        }
      }
    }
  }
})
