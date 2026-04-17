# Use an official Node runtime as a parent image
FROM node:20-slim

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the frontend code
COPY . .

# Expose the port the app runs on
EXPOSE 5173

# Command to run the application in dev mode with --host flag
CMD ["npm", "run", "dev", "--", "--host"]
