#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pydantic import BaseModel, Field

class MyFixItInstructionPairTemplate(BaseModel):
    """
    Schema for a FixIt instruction used with LangChain's `with_structured_output`.

    Fields:
        base_title: Original instruction title.
        base_step: Original instruction steps.
        base_component: Major component used in this base step.
        target_title: New instruction title.
        target_component: Major component that will be used in the target step.
    """
    base_title: str = Field(..., description="Original instruction title")
    base_steps: str = Field(..., description="All original instruction steps. Must include every step without omission.")
    base_component: str = Field(..., description="Major component used in these base steps")
    target_title: str = Field(..., description="New instruction title")
    target_component: str = Field(..., description="Major component that will be used in the new instruction steps")