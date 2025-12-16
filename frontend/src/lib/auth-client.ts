import { createAuthClient } from "better-auth/react"
import { stripeClient } from "@better-auth/stripe/client"

const baseURL = process.env.NEXT_PUBLIC_APP_URL || process.env.BETTER_AUTH_URL || "http://localhost:3000"

export const authClient = createAuthClient({
    baseURL,
    fetchOptions: {
        credentials: "include"
    },
    plugins: [
        stripeClient({
            subscription: true
        })
    ]
})
