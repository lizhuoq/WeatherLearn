import cdsapi

import asyncio
import os
from typing import Literal

Src_name = Literal["reanalysis-era5-pressure-levels", "reanalysis-era5-single-levels"]


async def request_wait(sleep, r):
    while True:
        r.update()
        reply = r.reply
        r.info("Request ID: %s, state: %s" % (reply["request_id"], reply["state"]))

        if reply["state"] == "completed":
            break
        elif reply["state"] in ("queued", "running"):
            r.info("Request ID: %s, sleep: %s", reply["request_id"], sleep)
            await asyncio.sleep(sleep)
        elif reply["state"] in ("failed",):
            r.error("Message: %s", reply["error"].get("message"))
            r.error("Reason:  %s", reply["error"].get("reason"))
            for n in (
                    reply.get("error", {}).get("context", {}).get("traceback", "").split("\n")
            ):
                if n.strip() == "":
                    break
                r.error("  %s", n)
            raise Exception(
                "%s. %s." % (reply["error"].get("message"), reply["error"].get("reason"))
            )


async def download(src_name: Src_name,
                   variable: list[str] | str,
                   year: list[str] | str,
                   month: list[str] | str,
                   day: list[str] | str,
                   time: list[str] | str,
                   out_path: str | None = None,
                   data_format: str = "netcdf",
                   product_type: str = "reanalysis",
                   sleep: int | None = None,
                   pressure_level: list[str] | str | None = None):
    """
    Download era5 reanalysis data by coroutine method
    """
    if os.path.exists(out_path):
        print(f"{out_path} is exists.")
        return

    c = cdsapi.Client(debug=True, wait_until_complete=False)

    request = {
        'product_type': product_type,
        'format': data_format,
        'variable': variable,
        'year': year,
        'month': month,
        'day': day,
        'time': time,
    }
    if pressure_level is not None:
        request["pressure_level"] = pressure_level

    r = c.retrieve(src_name, request)

    sleep = 30 if sleep is None else sleep
    await request_wait(sleep, r)

    r.download(out_path)
