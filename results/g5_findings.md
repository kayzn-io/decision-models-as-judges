A cascade asks the cheap judge first and only pays for the strong text model when the cheap judge is unsure.

The best cascade sets its confidence bar at 50%, reaches 55% accuracy at $0.02 per conversation, and sends 45% of conversations on to the strong text model. Sending every conversation to the strong text model reaches 54% accuracy at $0.03 per conversation, so the cascade costs 52% less while scoring about 1 point more. The fast text model on its own reaches 56% accuracy at $0.00 per conversation.

For someone choosing a judge, paying for the strong text model buys little here: the fast text model on its own is already about as accurate at $0.00 per conversation, so the cheaper judge is the better default.
