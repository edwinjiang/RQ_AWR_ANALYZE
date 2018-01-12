drop database if exists flask;
create database flask;
use flask;
drop table if exists entries;
create table entries (id int primary key auto_increment,title  varchar(10),content varchar(1000));