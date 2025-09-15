// Analytics utility functions
export const trackEvent = (eventName, parameters = {}) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', eventName, {
      event_category: 'engagement',
      ...parameters
    });
  }
};

export const trackPageView = (path, title) => {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('config', window.GA_MEASUREMENT_ID || '', {
      page_path: path,
      page_title: title
    });
  }
};

// Pre-defined event tracking functions
export const trackResumeDownload = (source = 'unknown') => {
  trackEvent('resume_download', {
    event_label: source,
    value: 1
  });
};

export const trackProjectView = (projectName, projectSlug) => {
  trackEvent('project_view', {
    event_label: projectName,
    custom_parameter_1: projectSlug
  });
};

export const trackContactFormSubmit = () => {
  trackEvent('contact_form_submit', {
    event_label: 'Contact Form',
    value: 1
  });
};

export const trackExternalLink = (url, linkText) => {
  trackEvent('outbound_link', {
    event_label: linkText,
    custom_parameter_1: url
  });
};

export const trackSocialLink = (platform, url) => {
  trackEvent('social_link_click', {
    event_label: platform,
    custom_parameter_1: url
  });
};