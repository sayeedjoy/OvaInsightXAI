import { NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000"

export async function POST(request: Request) {
    const payload = await request.json()

    if (!BACKEND_URL) {
        return NextResponse.json(
            { detail: "BACKEND_URL is not configured on the server." },
            { status: 500 }
        )
    }

    try {
        const response = await fetch(`${BACKEND_URL}/predict/pcos`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
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
        console.error("PCOS prediction proxy error:", error)
        return NextResponse.json(
            { detail: "Unable to reach prediction backend." },
            { status: 502 }
        )
    }
}

