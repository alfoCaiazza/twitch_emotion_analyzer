import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

function App() {
  const [emotions, setEmotions] = useState([]);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:6789');

    socket.onopen = () => {
      console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
      console.log('Emotion received:', event.data);
      const newEmotion = {
        time: new Date().getTime(), // Store as UNIX timestamp for easier handling
        type: event.data
      };
      setEmotions(prevEmotions => [...prevEmotions, newEmotion]);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return () => {
      socket.close();
    };
  }, []);

  // Generate the chart data by spreading emotions across their respective types.
  const chartData = emotions.reduce((acc, emotion) => {
    const lastEntry = acc.length ? acc[acc.length - 1] : {};
    const newEntry = {
      ...lastEntry,
      [emotion.type]: (lastEntry[emotion.type] || 0) + 1,
      time: emotion.time
    };
    acc.push(newEntry);
    return acc;
  }, []);

  return (
    <div>
      <LineChart width={1000} height={500} data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']}
               tickFormatter={(unixTime) => new Date(unixTime).toLocaleTimeString()} />
        <YAxis allowDecimals={false} />
        <Tooltip labelFormatter={(time) => new Date(time).toLocaleTimeString()} />
        <Legend />
        <Line type="monotone" dataKey="happy" stroke="#82ca9d" />
        <Line type="monotone" dataKey="sad" stroke="#8884d8" />
        <Line type="monotone" dataKey="angry" stroke="#dc143c" />
        <Line type="monotone" dataKey="fear" stroke="#ffa500" />
        <Line type="monotone" dataKey="surprise" stroke="#6a5acd" />
        <Line type="monotone" dataKey="neutral" stroke="#708090" />
        <Line type="monotone" dataKey="disgust" stroke="#006400" />
      </LineChart>
    </div>
  );
}

export default App;
