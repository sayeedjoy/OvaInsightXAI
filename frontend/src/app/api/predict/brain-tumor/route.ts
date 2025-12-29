import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000"

export async function POST(request: Request) {
    if (!BACKEND_URL) {
        return NextResponse.json(
            { detail: "BACKEND_URL is not configured on the server." },
            { status: 500 }
        )
    }

    try {
        // Get the form data from the request
        const formData = await request.formData()
        const file = formData.get("file") as File | null

        if (!file) {
            return NextResponse.json(
                { detail: "No file provided" },
                { status: 400 }
            )
        }

        // Create a new FormData to forward to backend
        const backendFormData = new FormData()
        backendFormData.append("file", file)

        const response = await fetch(`${BACKEND_URL}/predict/brain_tumor?include_xai=true`, {
            method: "POST",
            body: backendFormData
        })

        const data = await response.json().catch(() => null)

        if (!response.ok) {
            return NextResponse.json(
                data ?? { detail: "Prediction request failed." },
                { status: response.status }
            )
        }

        return NextResponse.json(data)
    } catch (error) {
        console.error("Brain tumor prediction proxy error:", error)
        return NextResponse.json(
            { detail: "Unable to reach prediction backend." },
            { status: 502 }
        )
    }
}

