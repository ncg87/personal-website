import React from 'react';
import { Helmet } from 'react-helmet-async';

const SEO = ({
  title = "Nickolas Goodis - Software Engineer & Data Scientist",
  description = "Software engineer specializing in React, TypeScript, Python, and machine learning. View my portfolio of full-stack applications, blockchain projects, and resume.",
  keywords = "software engineer, data scientist, React, TypeScript, Python, machine learning, blockchain, University of Miami",
  image = "/og-image.jpg",
  url = "https://nickogoodis.com",
  type = "website"
}) => {
  const fullTitle = title.includes("Nickolas Goodis") ? title : `${title} | Nickolas Goodis`;
  
  return (
    <Helmet>
      {/* Basic Meta Tags */}
      <title>{fullTitle}</title>
      <meta name="description" content={description} />
      <meta name="keywords" content={keywords} />
      <meta name="author" content="Nickolas Goodis" />
      
      {/* Open Graph / Facebook */}
      <meta property="og:type" content={type} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={image} />
      <meta property="og:url" content={url} />
      <meta property="og:site_name" content="Nickolas Goodis Portfolio" />
      
      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={image} />
      
      {/* Additional SEO */}
      <meta name="robots" content="index, follow" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="canonical" href={url} />
      
      {/* Schema.org markup for Google */}
      <script type="application/ld+json">
        {`
          {
            "@context": "https://schema.org",
            "@type": "Person",
            "name": "Nickolas Charles Goodis",
            "jobTitle": "Software Engineer",
            "description": "${description}",
            "url": "${url}",
            "image": "${image}",
            "alumniOf": {
              "@type": "CollegeOrUniversity",
              "name": "University of Miami"
            },
            "knowsAbout": [
              "Software Engineering",
              "Data Science", 
              "Machine Learning",
              "React",
              "TypeScript",
              "Python",
              "Blockchain"
            ],
            "sameAs": [
              "https://github.com/ncg87",
              "https://linkedin.com/in/nickogoodis"
            ]
          }
        `}
      </script>
    </Helmet>
  );
};

export default SEO;