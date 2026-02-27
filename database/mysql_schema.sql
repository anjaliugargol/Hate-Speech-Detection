-- Create Database
CREATE DATABASE IF NOT EXISTS Automated_hate_speech;

-- Use the Database
USE Automated_hate_speech;

-- Create Table to Store Hate Speech Data
CREATE TABLE IF NOT EXISTS HSD (
    CommentId VARCHAR(100),
    VideoId VARCHAR(100),
    Text TEXT,

    IsToxic TINYINT,
    IsAbusive TINYINT,
    IsThreat TINYINT,
    IsProvocative TINYINT,
    IsObscene TINYINT,
    IsHatespeech TINYINT,
    IsRacist TINYINT,
    IsNationalist TINYINT,
    IsSexist TINYINT,
    IsHomophobic TINYINT,
    IsReligiousHate TINYINT,
    IsRadicalism TINYINT
);

-- View Table Structure
DESCRIBE HSD;

-- Check inserted records
SELECT * FROM HSD LIMIT 10;
