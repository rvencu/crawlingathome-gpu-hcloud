create table jobs_en
(
    jobid    varchar(32)       not null
        constraint jobs_pk
            primary key,
    status   integer default 0 not null,
    modified timestamp
);

alter table jobs_en
    owner to cah;

create unique index jobs_jobid_uindex
    on jobs (jobid);

create trigger update_job_modtime
    before update
    on jobs_en
    for each row
execute procedure update_modified_column();

create table jobs_intl
(
    jobid    varchar(32)       not null
        constraint jobs_intl_pk
            primary key,
    status   integer default 0 not null,
    modified timestamp
);

alter table jobs_intl
    owner to cah;

create unique index jobs_intl_jobid_uindex
    on jobs_intl (jobid);

create trigger update_job_modtime
    before update
    on jobs_intl
    for each row
execute procedure update_modified_column();

create table jobs_nolang
(
    jobid    varchar(32)       not null
        constraint jobs_nolang_pk
            primary key,
    status   integer default 0 not null,
    modified timestamp
);

alter table jobs_nolang
    owner to cah;

create unique index jobs_nolang_jobid_uindex
    on jobs_nolang (jobid);

create trigger update_job_modtime
    before update
    on jobs_nolang
    for each row
execute procedure update_modified_column();

