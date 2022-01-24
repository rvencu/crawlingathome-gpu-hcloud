create table dataset_en
(
    sampleid bigint  not null
        constraint dataset_en_pk
            primary key,
    url      text    not null,
    text     text    not null,
    license  varchar,
    domain   varchar,
    wat      integer,
    status   smallint default 0,
    illegal  boolean  default false,
    hash     varchar not null,
    modified timestamp,
    language varchar not null,
    width    integer,
    height   integer
)
    with (autovacuum_analyze_threshold = 10000, autovacuum_vacuum_cost_limit = 50, autovacuum_vacuum_cost_delay = 0.1, autovacuum_vacuum_scale_factor = 0.1);

alter table dataset_en
    owner to cah;

create index dataset_en_status_index
    on dataset_en (status);

create trigger update_customer_modtime
    before update
    on dataset_en
    for each row
execute procedure update_modified_column();

create table dataset_intl
(
    sampleid bigint  not null
        constraint dataset_pk
            primary key,
    url      text    not null,
    text     text    not null,
    license  varchar,
    domain   varchar,
    wat      integer,
    status   smallint default 0,
    illegal  boolean  default false,
    hash     varchar not null,
    modified timestamp,
    language varchar not null,
    width    integer,
    height   integer
)
    with (autovacuum_analyze_threshold = 10000000, autovacuum_vacuum_cost_limit = 150, autovacuum_vacuum_cost_delay = 0.1, autovacuum_vacuum_scale_factor = 0);

alter table dataset_intl
    owner to cah;

create index dataset_status_index
    on dataset_intl (status);

create trigger update_customer_modtime
    before update
    on dataset_intl
    for each row
execute procedure update_modified_column();

create table dataset_nolang
(
    sampleid bigint  not null
        constraint dataset_nolang_pk
            primary key,
    url      text    not null,
    text     text    not null,
    license  varchar,
    domain   varchar,
    wat      integer,
    status   smallint default 0,
    illegal  boolean  default false,
    hash     varchar not null,
    modified timestamp,
    language varchar not null,
    width    integer,
    height   integer
)
    with (autovacuum_analyze_threshold = 10000000, autovacuum_vacuum_cost_limit = 150, autovacuum_vacuum_cost_delay = 0.1, autovacuum_vacuum_scale_factor = 0);

alter table dataset_nolang
    owner to cah;

create index dataset_nolang_status_index
    on dataset_nolang (status);

create trigger update_customer_modtime
    before update
    on dataset_nolang
    for each row
execute procedure update_modified_column();

create table dataset_buffer
(
    sampleid bigint,
    url      text    not null,
    text     text    not null,
    license  varchar,
    domain   varchar,
    wat      integer,
    status   smallint default 0,
    illegal  boolean  default false,
    hash     varchar not null,
    modified timestamp,
    language varchar not null,
    width    integer,
    height   integer
);

alter table dataset_buffer
    owner to cah;

create trigger skip_errors
    before insert
    on dataset_buffer
    for each row
execute procedure on_insert_in_original_table();

