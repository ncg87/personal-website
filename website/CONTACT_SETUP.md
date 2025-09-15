# Contact Form Backend Setup

The contact form supports multiple backend services for handling form submissions. Here are the recommended options:

## Option 1: Formspree (Recommended)

1. Sign up at [formspree.io](https://formspree.io)
2. Create a new form and get your form endpoint
3. Add to your `.env` file:
   ```
   VITE_CONTACT_FORM_ENDPOINT=https://formspree.io/f/your-form-id
   ```

**Pros:**
- Easy setup
- Free tier available
- Spam protection
- Email notifications
- Form data dashboard

## Option 2: Netlify Forms

1. Add `netlify` attribute to your form in the JSX
2. Deploy to Netlify
3. Forms will be automatically handled

**Pros:**
- No external service needed if using Netlify
- Integrated with site deployment
- Built-in spam protection

## Option 3: Custom Backend

If you prefer a custom solution, the form sends a JSON payload with:

```json
{
  "name": "User Name",
  "email": "user@example.com", 
  "subject": "Contact Subject",
  "message": "User message",
  "_replyto": "user@example.com"
}
```

Set your endpoint in the environment variable:
```
VITE_CONTACT_FORM_ENDPOINT=https://your-api.com/contact
```

## Fallback Behavior

If no backend is configured, the form will automatically fall back to a `mailto:` link, ensuring the contact form always works.

## Setup Instructions

1. Copy `.env.example` to `.env`
2. Configure your chosen backend service
3. Update the environment variables
4. Test the form to ensure it works correctly

## Analytics Integration

The form automatically tracks submissions via Google Analytics when properly configured. Events are sent with:
- Event name: `contact_form_submit`
- Category: `engagement` 
- Label: `Contact Form`