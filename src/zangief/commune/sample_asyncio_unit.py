
"""

Generic unit test using pytest that can be written using the asyncio client/interface logic

Note: run this install command: `pip install pytest-asyncio pytest-mock`

import pytest
from unittest.mock import AsyncMock
import asyncio
from .comx import AsyncIOClient

@pytest.mark.asyncio
async def test_run_async_task(mocker):
    mock_coroutine = AsyncMock()
    mock_coroutine.return_value = "Mocked task complete!"

    mock_run = mocker.patch("asyncio.run", return_value=mock_coroutine())

    client = AsyncIOClient()
    result = await client.run(main=mock_coroutine)

    mock_run.assert_called_once_with(main=mock_coroutine)

    assert result == "Mocked task complete!"

"""