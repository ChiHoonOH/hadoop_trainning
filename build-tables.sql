DROP table tweets_raw;
create external table tweets_raw (
  `polarity` int,
  `id` string,
  `date` string,
  `query` string,
  `user` string,
  `text` string )
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
stored as textfile
location '/sentiment' 
tblproperties("skip.header.line.count"="0");

drop table tweets;
create table tweets stored as ORC as select * from tweets_raw;

drop table sentiment_words;
create external table sentiment_words (
  `word`  string,
  `score` int  )
row format delimited
fields terminated by '\t'
stored as textfile
location '/data';

