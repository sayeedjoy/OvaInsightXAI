const site_url = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";

export const site = {
  name: "OvaInsightXAI",
  description: "Advanced ovarian cancer prediction using AI and machine learning",
  url: site_url,
  ogImage: `${site_url}/og.jpg`,
  logo: "/logo.svg",
  mailSupport: "hello@domain.com", // Support email address
  mailFrom: process.env.MAIL_FROM || "noreply@domain.com", // Transactional email address
  links: {
    twitter: "https://twitter.com/sayeedjoy",
    github: "https://github.com/sayeedjoy",
    linkedin: "https://www.linkedin.com/in/sayeedjoy",
  }
} as const;