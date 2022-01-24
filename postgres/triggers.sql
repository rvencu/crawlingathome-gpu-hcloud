create user cah
    superuser
    createdb
    createrole;


create function update_modified_column() returns trigger
    language plpgsql
as
$$
BEGIN
    NEW.modified = now();
    RETURN NEW;
END;
$$;

alter function update_modified_column() owner to cah;

create function on_insert_in_original_table() returns trigger
    language plpgsql
as
$$
BEGIN
    BEGIN
        IF NEW.language = 'en' THEN
            INSERT INTO dataset_en (sampleid, url, text, license, domain, wat, hash, modified, language, width, height)
            VALUES (NEW.sampleid, NEW.url, NEW.text, NEW.license, NEW.domain, NEW.wat, NEW.hash, NEW.modified, NEW.language,
                    NEW.width, NEW.height)
            ON CONFLICT DO NOTHING;
        ELSIF NEW.language = '' THEN
            INSERT INTO dataset_nolang (sampleid, url, text, license, domain, wat, hash, modified, language, width, height)
            VALUES (NEW.sampleid, NEW.url, NEW.text, NEW.license, NEW.domain, NEW.wat, NEW.hash, NEW.modified, NEW.language,
                    NEW.width, NEW.height)
            ON CONFLICT DO NOTHING;
        ELSE
            INSERT INTO dataset_intl (sampleid, url, text, license, domain, wat, hash, modified, language, width, height)
            VALUES (NEW.sampleid, NEW.url, NEW.text, NEW.license, NEW.domain, NEW.wat, NEW.hash, NEW.modified, NEW.language,
                    NEW.width, NEW.height)
            ON CONFLICT DO NOTHING;
        END IF;
    EXCEPTION
        WHEN OTHERS THEN
            NULL;
    END;
RETURN NULL;
END;
$$;

alter function on_insert_in_original_table() owner to cah;

